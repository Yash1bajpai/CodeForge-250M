"""Dual-T4 ordering and collective primitives; torch is imported only when used."""
import math
from contextlib import nullcontext
import random

GLOBAL_BATCH = 256
WORLD_SIZE = 2


def geometry(batch_size, single_accumulation, world_size, rank):
    if (world_size != WORLD_SIZE or rank not in (0, 1)
            or type(batch_size) is not int or batch_size < 1
            or type(single_accumulation) is not int or single_accumulation % WORLD_SIZE
            or batch_size * single_accumulation != GLOBAL_BATCH):
        raise ValueError("Dual T4 requires world2 and exact global batch256 with even accumulation")
    return single_accumulation // WORLD_SIZE


def require_hardware(names, world_size, rank, local_rank):
    if (world_size != WORLD_SIZE or rank not in (0, 1) or local_rank != rank
            or len(names) != WORLD_SIZE or any(name not in ("Tesla T4", "NVIDIA T4") for name in names)):
        raise ValueError("Dual T4 requires one node, two processes and exactly two visible T4s")


class RankIndices:
    """Shard whole ORIGINAL microbatches, not rows: padded-token loss means depend on grouping."""
    def __init__(self, indices, batch_size, rank):
        self.indices, self.batch_size, self.rank = indices, batch_size, rank

    def __iter__(self):
        for start in range(self.rank * self.batch_size, len(self.indices), WORLD_SIZE * self.batch_size):
            for offset in range(start, min(start + self.batch_size, len(self.indices))):
                yield self.indices[offset]

    def __len__(self):
        blocks, tail = divmod(len(self.indices), self.batch_size)
        full = (blocks + WORLD_SIZE - 1 - self.rank) // WORLD_SIZE
        return full * self.batch_size + (tail if blocks % WORLD_SIZE == self.rank else 0)


def local_rows(global_rows, batch_size, rank, micro_step):
    return max(0, min(batch_size, global_rows - (WORLD_SIZE * micro_step + rank) * batch_size))


def rank_zero(action, device):
    """Only rank0 performs effects; propagate success OR failure before any next collective."""
    import torch.distributed as dist
    result = [None]
    if dist.get_rank() == 0:
        try:
            result[0] = (True, action())
        except Exception as error:
            result[0] = (False, type(error).__name__)
    dist.broadcast_object_list(result, src=0, device=device)
    ok, value = result[0]
    if not ok:
        raise RuntimeError("Rank0 action failed: " + value)
    return value


def step_totals(loss_sum, rows, tokens, device):
    import torch
    import torch.distributed as dist
    totals = torch.tensor([loss_sum, rows, tokens], dtype=torch.float64, device=device)
    dist.all_reduce(totals)
    loss, rows, tokens = totals.tolist()
    if rows <= 0:
        raise ValueError("No real rows in a distributed update")
    return loss / rows, int(rows), int(tokens)


def any_rank(value, device):
    import torch
    import torch.distributed as dist
    flag = torch.tensor(int(bool(value)), device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def backward(model, batches, global_rows, batch_size, rank, sequence_length, scaler, device, dtype):
    """Same microbatch means as single GPU, including padding and an odd final tail."""
    import torch
    if not 0 < global_rows <= GLOBAL_BATCH:
        raise ValueError("Distributed update must consume1to256 planned real rows")
    micro_steps = math.ceil(global_rows / (batch_size * WORLD_SIZE))
    loss_sum, rows, tokens = 0.0, 0, 0
    for micro_step in range(micro_steps):
        expected = local_rows(global_rows, batch_size, rank, micro_step)
        if expected:
            try:
                x, y, counts = next(batches)
            except StopIteration as error:
                raise RuntimeError("Rank input ended before its planned global prefix") from error
            if x.shape[0] != expected:
                raise RuntimeError("Rank grouping differs from the original single-GPU microbatch")
            tokens += int(counts.sum())
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        else:
            # Synthetic zeros contribute zero loss/rows/tokens, not a repeated corpus row.
            x = torch.zeros((1, sequence_length - 1), dtype=torch.long, device=device)
            y = torch.zeros_like(x)
        sync = model.no_sync() if micro_step + 1 < micro_steps else nullcontext()
        with sync:
            with torch.amp.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                _, loss = model(x, y)
                weighted = loss * (expected * WORLD_SIZE / GLOBAL_BATCH)
            scaler.scale(weighted).backward()
        if expected:
            loss_sum += loss.item() * expected
            rows += expected
    loss, rows, tokens = step_totals(loss_sum, rows, tokens, device)
    if rows != global_rows:
        raise RuntimeError("Consumed rows differ from the exact planned prefix")
    if rows < GLOBAL_BATCH:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(GLOBAL_BATCH / rows)
    return loss, rows, tokens


def amp_step(scaler, optimizer, gradient_norm, device):
    """Unscale/clip already done. All ranks step OR skip, with identical public scaler state."""
    import torch.distributed as dist
    state = scaler.state_dict()
    states = [None] * WORLD_SIZE
    dist.all_gather_object(states, state)
    if any(item != state for item in states):
        raise RuntimeError("Loss scaler states differ across ranks; no optimizer update")
    overflow = any_rank(not math.isfinite(float(gradient_norm)), device)
    if overflow:
        if not scaler.is_enabled():
            raise FloatingPointError("Nonfinite distributed gradients without loss scaling")
        state["scale"] *= state["backoff_factor"]
        state["_growth_tracker"] = 0
        scaler.update(new_scale=state["scale"])
        scaler.load_state_dict(state)
        return False
    scaler.step(optimizer)
    scaler.update()
    return True


def checkpoint_rng(device):
    import torch
    import torch.distributed as dist
    local = {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
             "torch_cuda": [torch.cuda.get_rng_state(device)] if device.type == "cuda" else []}
    ranks = [None] * WORLD_SIZE
    dist.all_gather_object(ranks, local)
    return {**ranks[0], "distributed": {"world_size": WORLD_SIZE, "ranks": ranks}}


def restore_rng(saved, rank, device, distributed):
    import torch
    if distributed and "distributed" in saved:
        group = saved["distributed"]
        if group.get("world_size") != WORLD_SIZE or len(group.get("ranks", [])) != WORLD_SIZE:
            raise ValueError("Checkpoint distributed RNG geometry differs")
        saved = group["ranks"][rank]
    random.setstate(saved["python"])
    torch.set_rng_state(saved["torch_cpu"])
    if device.type == "cuda" and saved.get("torch_cuda"):
        # Restore only this process's device; never initialize the peer GPU's RNG.
        torch.cuda.set_rng_state(saved["torch_cuda"][0], device=device)


def validation(guard, step, evaluate, device):
    """Use exactly the single-rank validation batches, then the SAME reading/history on both ranks."""
    loss = rank_zero(evaluate, device)
    fired = guard.record(step, loss)
    return loss, fired
