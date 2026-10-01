import os
import sys
import time
import math
import glob
import json
import random
import traceback
import threading
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.architecture import CodeForgeModel
from models.init_weights import init_model_weights
from training.utils import get_lr_cosine_schedule
from training.resume_safety import audit_shards, manifest_fingerprint, optimizer_step_counts, validate_resume_metadata, file_sha256, migrated_permutation

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.logfile = open(filepath, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

class LazyShardDataset(Dataset):
    """
    B4 fix: lazy per-shard loading. Only ONE shard (~1000 seqs) is in RAM at a
    time per worker. Shards are read-only uint16 tensors on disk.
    Single-pass by design: the trainer consumes a shuffled shard order and
    NEVER wraps around (epoch guard lives in the training loop).
    """
    def __init__(self, tokenized_dir: str, seq_length: int = 2048, shard_files=None, vocab_size=32000):
        self.seq_length = seq_length
        self.shard_files = shard_files if shard_files is not None else sorted(
            glob.glob(os.path.join(tokenized_dir, "shard_*.pt")))
        if not self.shard_files:
            raise SystemExit("ERROR: no shards in data/tokenized. Run the data pipeline first.")
        self.audit_records = audit_shards(self.shard_files, seq_length, vocab_size, allow_partial=True)
        self.rows = [r['rows'] for r in self.audit_records]
        self.offsets = [0]
        for n in self.rows:
            self.offsets.append(self.offsets[-1]+n)
        total = self.offsets[-1]
        self.data = torch.empty(total, seq_length, dtype=torch.uint16)
        for i,path in enumerate(self.shard_files):
            t = torch.load(path,map_location='cpu',weights_only=True)
            self.data[self.offsets[i]:self.offsets[i+1]] = t
        print(f'--> [AuditedData] {len(self.shard_files)} shards, {total} real rows; no padding')
    def __len__(self):
        return self.offsets[-1]
    def __getitem__(self, idx):
        seq = self.data[idx].long()  # uint16 -> long for embedding lookup
        x = seq[:-1]
        y = seq[1:]
        return x, y

def verify_tokenizer(tokenizer_dir: str):
    """B3 fix: hard-fail if any of the 11 special tokens are missing."""
    REQUIRED = ["<|endoftext|>", "<|unk|>", "<|pad|>", "<|fim_prefix|>", "<|fim_middle|>",
                "<|fim_suffix|>", "<|tool_call|>", "<|tool_result|>", "<|thinking|>",
                "<|json_start|>", "<|json_end|>"]
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    missing = [t for t in REQUIRED if t not in tok.get_vocab()]
    if missing:
        raise SystemExit(f"ERROR: tokenizer missing special tokens {missing}. "
                         "Retrain tokenizer on the dedup corpus (train_tokenizer.py).")
    print(f"--> [Tokenizer] All {len(REQUIRED)} special tokens present (B3 verified).")
    return tok

def evaluate_val_loss(model, val_batches, device, dtype):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in val_batches:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=dtype):
                _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))

_METRICS_LOCK = threading.Lock()  # telemetry thread reads metrics.json concurrently

def append_metric(metrics_path, entry):
    """Watchdog food: append one JSON line to metrics.json (pulled by watch_shift.sh)."""
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with _METRICS_LOCK:
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

def save_checkpoint(raw_model, optimizer, step, loss_val, val_loss, ckpt_dir, latest_path, keep=2, resume_state=None, scaler=None):
    state = {
        'step': step,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_val,
        'val_loss': val_loss,
        'resume_state': resume_state,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else {},
        'rng_state': {
            'python': random.getstate(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    # atomic write: a kill/OOM mid-save must never leave a truncated
    # latest_checkpoint.pt (it propagates to the shared HF rotation repo and
    # bricks every account's next resume)
    tmp_path = latest_path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, latest_path)
    step_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
    torch.save(state, step_path)
    # B5 fix: rotate — keep only the newest `keep` step checkpoints
    old_steps = sorted(
        int(os.path.basename(p).split("_")[-1].split(".")[0])
        for p in glob.glob(os.path.join(ckpt_dir, "checkpoint_step_*.pt")))
    for s in old_steps[:-keep]:
        try:
            os.remove(os.path.join(ckpt_dir, f"checkpoint_step_{s}.pt"))
        except OSError:
            pass
    return step_path

def train():
    parser = argparse.ArgumentParser(description="CodeForge-250M Training Engine (run #2)")
    parser.add_argument("--from_scratch", action="store_true", help="Force step 0 with fresh init")
    parser.add_argument("--resume", action="store_true", help="Explicitly resume from latest_checkpoint.pt (B2: no silent resumes)")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs/config_250M.yaml"))
    parser.add_argument("--max_hours", type=float, default=11.0, help="Wall-clock budget before clean stop (Kaggle ~12h limit)")
    parser.add_argument("--session-steps", type=int, default=None, help="Bounded smoke test; does not change LR schedule")
    parser.add_argument("--init_from", type=str, default=None, help="Warm start: load MODEL WEIGHTS ONLY from this checkpoint (step 0, fresh optimizer/schedule/data cursor). Never overwrites the source file.")
    parser.add_argument("--legacy-migration", type=str, help="Audited JSON identity/counter record for one legacy checkpoint")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise SystemExit("This audited Stage A resume supports only one GPU.")

    # --- DDP setup (Kaggle T4 x2): env vars set by torchrun ---
    ddp_world = int(os.environ.get("WORLD_SIZE", "1"))
    ddp_rank = int(os.environ.get("RANK", "0"))
    ddp_local = int(os.environ.get("LOCAL_RANK", "0"))
    is_ddp = ddp_world > 1
    if is_ddp:
        import torch.distributed as dist
        torch.cuda.set_device(ddp_local)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{ddp_local}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_main = (ddp_rank == 0)

    log_path = os.path.join(PROJECT_ROOT, "training.log")
    if is_main:
        sys.stdout = Logger(log_path)
        sys.stderr = Logger(log_path)   # tracebacks must reach training.log too

    mode = f"DDP x{ddp_world}" if is_ddp else "SINGLE"
    print(f"=== [CODEFORGE-250M RUN #2 — FRESH, SINGLE-PASS, WATCHED — {mode}] ===")

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is not available!")
        return

    cap = torch.cuda.get_device_capability(0)
    dtype = torch.float16 if cap[0] < 8 else torch.bfloat16
    print(f"--> Hardware: {torch.cuda.get_device_name(0)} | dtype={dtype} | world={ddp_world}")

    with open(args.config, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
        cfg = full_cfg["model"]
        train_cfg = full_cfg["training"]

    metrics_path = os.path.join(PROJECT_ROOT, "metrics.json")
    ckpt_dir = os.path.join(PROJECT_ROOT, full_cfg.get("checkpointing", {}).get("output_dir", "checkpoints/CodeForge-250M"))
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = os.path.join(ckpt_dir, "latest_checkpoint.pt")

    # LIVE TELEMETRY: push metrics + log tail to a tiny Kaggle dataset every
    # 15 min so the outside-world supervisor (agy) can watch the run while the
    # session is still going. Failures must NEVER touch training.
    # (_METRICS_LOCK is module-level; telemetry thread reads metrics.json.)
    _TELEMETRY_STOP = threading.Event()

    def _telemetry_uploader():
        import subprocess as _sp, shutil as _sh
        tdir = "/tmp/telemetry_push"
        while not _TELEMETRY_STOP.wait(900):
            try:
                os.makedirs(tdir, exist_ok=True)
                with open(f"{tdir}/dataset-metadata.json", "w") as f:
                    json.dump({"title": "codeforge-telemetry",
                               "id": "yashbajpai2027/codeforge-telemetry",
                               "isPrivate": True,
                               "licenses": [{"name": "CC0-1.0"}]}, f)
                if os.path.exists(metrics_path):
                    with _METRICS_LOCK:
                        _sh.copy(metrics_path, tdir)
                lp = os.path.join(PROJECT_ROOT, "training.log")
                if os.path.exists(lp):
                    with open(lp, "r", errors="replace") as f:
                        tail = f.readlines()[-400:]
                    with open(f"{tdir}/training_tail.log", "w") as f:
                        f.writelines(tail)
                step_now = 0
                try:
                    with _METRICS_LOCK:
                        with open(metrics_path) as f:
                            step_now = max((json.loads(l).get("step", 0)
                                            for l in f if l.strip()), default=0)
                except Exception:
                    pass
                with open(f"{tdir}/step.json", "w") as f:
                    json.dump({"step": step_now, "ts": time.time()}, f)
                r = _sp.run(["kaggle", "datasets", "version", "-p", tdir,
                             "-m", "live", "--dir-mode", "zip"],
                            capture_output=True, text=True, timeout=180)
                print(f"[Telemetry] push rc={r.returncode} step={step_now}", flush=True)
            except Exception as e:
                print(f"[Telemetry] push failed (ignored): {e}", flush=True)

    _telemetry_thread = None
    if is_main and not os.environ.get("CF_TELEMETRY_DISABLED"):
        _telemetry_thread = threading.Thread(target=_telemetry_uploader, daemon=True)
        _telemetry_thread.start()

    batch_size_per_device = train_cfg.get("batch_size_per_device", 16)
    accum_steps = train_cfg.get("gradient_accumulation_steps", 16)
    # DDP: per-device batch x world must give the SAME total tokens/step (524,288).
    # single: 16 x 16 accum = 256 seqs. 2x T4: 16 x 8 accum x 2 gpus = 256 seqs.
    if is_ddp:
        accum_steps = max(1, accum_steps // ddp_world)
    max_steps = train_cfg.get("max_steps", 4482)
    max_lr = float(train_cfg.get("learning_rate", 6.0e-4))
    min_lr = float(train_cfg.get("min_learning_rate", 6.0e-5))
    warmup_steps = int(train_cfg.get("warmup_steps", 500))
    save_steps = int(full_cfg.get("checkpointing", {}).get("save_steps", 1000))
    save_interval_sec = int(full_cfg.get("checkpointing", {}).get("save_time_interval_minutes", 20)) * 60
    val_steps = int(train_cfg.get("val_steps", 100))          # B7: val eval cadence
    val_batches_n = int(train_cfg.get("val_batches", 4))
    seq_len = cfg["max_position_embeddings"]

    # B3: tokenizer must be complete before anything trains
    tok = verify_tokenizer(os.path.join(PROJECT_ROOT, "data/tokenizer"))
    if len(tok) != cfg['vocab_size']:
        raise SystemExit('Tokenizer size differs from checkpoint/model config')

    # B7: hold out a deterministic slice of shards as the validation set
    all_shards = sorted(glob.glob(os.path.join(PROJECT_ROOT, "data/tokenized/shard_*.pt")))
    if not all_shards:
        raise SystemExit("ERROR: data/tokenized is empty. Run the Kaggle data-prep kernel first.")
    rng = random.Random(42)
    rng.shuffle(all_shards)
    n_val = max(1, int(len(all_shards) * 0.005))
    val_shards, train_shards = all_shards[:n_val], all_shards[n_val:]
    print(f"--> [B7] {len(train_shards)} train shards / {len(val_shards)} val shards (0.5% held out)")

    model = CodeForgeModel(cfg).to(device)
    # T4 VRAM: recompute activations in backward (~10x activation reduction)
    model.gradient_checkpointing = True
    print(f"--> Parameters: {model.get_parameter_count():,} | grad_checkpointing=True")

    # DDP wrap: gradient sync across T4s. device_ids=[local], single process per GPU.
    if is_ddp:
        import torch.distributed as dist
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local])
        raw_model = model.module
    else:
        raw_model = model

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr,
                                  weight_decay=train_cfg.get("weight_decay", 0.1),
                                  betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))

    # --- B2: explicit resume only ---
    start_step = 0
    checkpoint_state = None
    resume_state = None
    loss_val, val_loss = 0.0, None
    if args.resume:
        if not os.path.exists(latest_path):
            raise SystemExit("ERROR: --resume passed but no latest_checkpoint.pt found.")
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=True)  # cpu: avoid GPU memory spike during restore
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt.get('step', 0)
        loss_val = ckpt.get('loss', 0.0)
        val_loss = ckpt.get('val_loss', None)
        checkpoint_state = ckpt
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"--> [Resume] Explicitly resumed from step {start_step} (loss {loss_val:.4f})")
    elif args.init_from:
        if os.path.exists(latest_path):
            raise SystemExit("ERROR: --init_from needs an empty checkpoint dir (latest_checkpoint.pt exists).")
        _src = torch.load(args.init_from, map_location="cpu", weights_only=True)
        raw_model.load_state_dict(_src['model_state_dict'])
        print(f"--> [Warm start] weights only from {args.init_from} (source step {_src.get('step')}, "
              f"val_loss {_src.get('val_loss')}); optimizer, LR schedule and data cursor start fresh", flush=True)
        del _src
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif os.path.exists(latest_path) and not args.from_scratch:
        raise SystemExit("ERROR: latest_checkpoint.pt exists. Pass --resume to continue it, "
                         "or --from_scratch to discard it. Silent resume is forbidden (B2).")
    else:
        if args.from_scratch and os.path.exists(latest_path):
            os.remove(latest_path)
        init_model_weights(raw_model, initializer_range=0.02)
        print("--> [From Scratch] Fresh random init at step 0")

    if args.resume and checkpoint_state.get('resume_state') is None:
        if not args.legacy_migration:
            raise SystemExit('Legacy checkpoint: an audited migration record is required before resume.')
        with open(args.legacy_migration) as f:
            migration = json.load(f)
        if migration.get('checkpoint_sha256') != file_sha256(latest_path):
            raise SystemExit('Legacy checkpoint hash differs from audited migration.')
        if migration.get('reported_step') != checkpoint_state.get('step'):
            raise SystemExit('Legacy checkpoint counter differs from audited migration.')
        checkpoint_state['resume_state'] = migration['resume_state']
        start_step = migration['schedule_step']
        if migration['resume_state']['legacy_sample_cursor'] != start_step * batch_size_per_device * accum_steps:
            raise SystemExit('Audited legacy cursor and schedule do not agree')
        print('--> [Migration] Legacy RNG/scaler history unavailable; exact bitwise replay is not claimed.')

    # --- datasets: single-pass over train shards (no while-True wrap!) ---
    train_dataset = LazyShardDataset(os.path.join(PROJECT_ROOT, "data/tokenized"),
                                     seq_length=seq_len, shard_files=train_shards, vocab_size=cfg["vocab_size"])
    val_dataset = LazyShardDataset(os.path.join(PROJECT_ROOT, "data/tokenized"),
                                   seq_length=seq_len, shard_files=val_shards, vocab_size=cfg["vocab_size"])
    all_records = sorted(train_dataset.audit_records + val_dataset.audit_records, key=lambda x:x['name'])
    fingerprint = manifest_fingerprint(all_records, os.path.join(PROJECT_ROOT, 'data/tokenizer/tokenizer.json'))
    seqs_per_step = batch_size_per_device * accum_steps * ddp_world
    if checkpoint_state is not None:
        resume_state = validate_resume_metadata(checkpoint_state, fingerprint, seqs_per_step)
        if resume_state.get('order_version') != 'legacy-filtered-v1':
            raise SystemExit('Unknown saved permutation identity')
        rng_state = checkpoint_state.get('rng_state')
        if rng_state:
            random.setstate(rng_state['python'])
            torch.set_rng_state(rng_state['torch_cpu'])
            if torch.cuda.is_available() and rng_state.get('torch_cuda'):
                torch.cuda.set_rng_state_all(rng_state['torch_cuda'])
    else:
        resume_state = {'schema_version':1, 'fingerprint':fingerprint, 'seqs_per_step':seqs_per_step,
                        'sample_cursor':0, 'completed_updates':0, 'optimizer_step_count':0, 'order_version':'legacy-filtered-v1'}
    if is_ddp:
        import torch.distributed as dist
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=ddp_world, rank=ddp_rank, shuffle=True, drop_last=True)
        dataloader = DataLoader(train_dataset, batch_size=batch_size_per_device,
                                sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True)
    else:
        train_sampler = None
        # Deterministic global permutation (fixed seed): every session sees the
        # SAME order. On resume we start the sampler at the exact sample offset,
        # so the single pass continues where it stopped — no re-shuffle, no data
        # cycling (run-#1 killer #2).
        _perm, _valid = migrated_permutation(train_dataset.rows)
        if resume_state.get('legacy_sample_cursor') is not None:
            expected_cursor = int(_valid[:resume_state['legacy_sample_cursor']].sum())
            if expected_cursor != resume_state['sample_cursor']:
                raise SystemExit('Migration cursor does not match filtered legacy permutation')

        class _OrderedIndices(torch.utils.data.Sampler):
            def __init__(self, indices):
                self.indices = indices
            def __iter__(self):
                return iter(self.indices)
            def __len__(self):
                return len(self.indices)

        skip_samples = resume_state['sample_cursor']
        if skip_samples > len(_perm):
            raise SystemExit('Invalid saved sample cursor')
        if skip_samples:
            print(f"--> [Resume] data position: {skip_samples:,}/{len(_perm):,} samples already "
                  f"consumed — continuing single pass from there", flush=True)
        dataloader = DataLoader(train_dataset, batch_size=batch_size_per_device,
                                sampler=_OrderedIndices(_perm[skip_samples:]),
                                num_workers=2, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_per_device, shuffle=False,
                            num_workers=1, pin_memory=True, drop_last=True)

    del checkpoint_state
    val_batches = []
    for i, b in enumerate(val_loader):
        val_batches.append(b)
        if i + 1 >= val_batches_n:
            break

    total_train_seqs = len(train_dataset)
    seqs_per_step = batch_size_per_device * accum_steps * ddp_world
    if total_train_seqs < seqs_per_step * max_steps and is_main:
        print(f"WARNING: corpus holds {total_train_seqs:,} seqs but {max_steps} steps x {seqs_per_step} seqs/step "
              f"= {seqs_per_step * max_steps:,} needed. Single-pass will fall short — trim max_steps or add data.")

    model.train()
    start_time = time.time()
    tokens_seen = 0
    optimizer.zero_grad()

    # WATCHDOG STATE
    val_history = []
    loss_floor_strikes = 0
    last_save_time = time.time()

    def watchdog_write(reason, extra=None):
        entry = {"event": "STOP", "reason": reason, "step": step, "ts": time.time()}
        if extra:
            entry.update(extra)
        append_metric(metrics_path, entry)
        print(f"--> [WATCHDOG] {reason} at step {step}. Saving and exiting.", flush=True)

    # SINGLE-PASS ITERATOR: StopIteration ends the run (epoch guard — run #1 fix)
    batch_generator = iter(dataloader)
    steps_planned = max_steps - start_step
    if args.session_steps is not None:
        steps_planned = min(steps_planned,args.session_steps)

    step = start_step
    epoch_done = False
    crashed = False
    try:
      try:
        for step_offset in range(1, steps_planned + 1):
            next_step = start_step + step_offset

            # clock guard: clean stop before the wall kills us (rank0 decides,
            # broadcasts; both ranks break together to avoid NCCL desync)
            wall_hit = ((time.time() - start_time) / 3600.0 >= args.max_hours
                        or time.time() >= float(os.environ.get('CF_ABSOLUTE_STOP_EPOCH', 'inf')))
            if is_ddp:
                t = torch.tensor([1 if wall_hit else 0], device=device)
                torch.distributed.all_reduce(t)
                wall_hit = t.item() > 0
            if wall_hit:
                if is_main:
                    watchdog_write("WALL_CLOCK_BUDGET")
                break

            current_lr = get_lr_cosine_schedule(next_step, warmup_steps, max_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            step_loss = 0.0
            batch_seqs = 0
            for micro_step in range(accum_steps):
                try:
                    x, y = next(batch_generator)
                except StopIteration:
                    epoch_done = True
                    batch_generator = None
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=dtype):
                    logits, loss = model(x, y)
                    scaled_loss = loss * (x.shape[0] / seqs_per_step)
                scaler.scale(scaled_loss).backward()
                step_loss += loss.item() * x.shape[0]
                batch_seqs += x.shape[0]
                # GLOBAL token count: per-rank numel x world (x is seq[:-1] = 2047 tok)
                tokens_seen += x.numel() * ddp_world
            epoch_done = (batch_generator is None)
            if is_ddp:
                t = torch.tensor([1 if epoch_done else 0], device=device)
                torch.distributed.all_reduce(t)
                epoch_done = t.item() > 0
            if epoch_done and batch_seqs == 0:
                if is_main:
                    watchdog_write("EPOCH_COMPLETE_SINGLE_PASS")
                break

            if batch_seqs < seqs_per_step:
                # Normalize the final partial batch by its actual example count.
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(seqs_per_step / batch_seqs)
            step_loss /= batch_seqs
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.get("max_grad_norm", 1.0))
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            update_applied = scaler.get_scale() >= previous_scale
            step = next_step
            resume_state['sample_cursor'] += batch_seqs
            resume_state.pop('legacy_sample_cursor', None)
            resume_state['completed_updates'] += int(update_applied)
            resume_state['optimizer_step_count'] = optimizer_step_counts(optimizer.state_dict())
            optimizer.zero_grad()
            loss_val = step_loss

            # TOKEN ACCOUNTANT (run #1 killer #1): real tokens/step must match config.
            # x carries seq[:-1] (2047 of 2048 tokens) — 0.05% under is expected.
            expected_tokens_per_step = seqs_per_step * seq_len * (1 - 1.0 / seq_len)
            actual = tokens_seen / step_offset
            mismatch = not epoch_done and step_offset >= 5 and abs(actual - expected_tokens_per_step) > 0.02 * expected_tokens_per_step
            if is_ddp:
                t = torch.tensor([1 if mismatch else 0], device=device)
                torch.distributed.all_reduce(t)
                mismatch = t.item() > 0
            if mismatch:
                if is_main:
                    watchdog_write("TOKEN_COUNT_MISMATCH",
                                   {"expected": expected_tokens_per_step, "actual": actual})
                break

            # LOSS FLOOR (run #1 killer #3): ppl < 1.5 mid-run = memorization signature
            if loss_val < math.log(1.5):
                loss_floor_strikes += 1
                floor_hit = loss_floor_strikes >= 3
            else:
                loss_floor_strikes = 0
                floor_hit = False
            if is_ddp:
                t = torch.tensor([1 if floor_hit else 0], device=device)
                torch.distributed.all_reduce(t)
                floor_hit = t.item() > 0
            if floor_hit:
                if is_main:
                    watchdog_write("LOSS_FLOOR_MEMORIZATION", {"loss": loss_val})
                break

            if step % 5 == 0 or step_offset == 1:
                if is_main:
                    ppl = math.exp(min(loss_val, 20.0))
                    vram_gb = torch.cuda.memory_allocated() / (1024**3)
                    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
                    elapsed = time.time() - start_time
                    print(f"{step:<6} | {loss_val:<8.4f} | {ppl:<10.2f} | {current_lr:<10.2e} | {vram_gb:<6.2f}/{peak_gb:<6.2f} | {elapsed/60:.0f}m", flush=True)
                    append_metric(metrics_path, {
                        "event": "TRAIN", "step": step, "loss": round(loss_val, 4),
                        "ppl": round(ppl, 2), "lr": current_lr, "tokens_seen": tokens_seen,
                        "ts": time.time()})

            # B7: validation cadence + divergence kill. BOTH ranks evaluate (keeps
            # collectives symmetric); rank0 decides, broadcasts, both break together.
            val_diverged = False
            if step % val_steps == 0:
                val_loss = evaluate_val_loss(raw_model, val_batches, device, dtype)
                if is_main:
                    val_history.append(val_loss)
                    print(f"--> [VAL] step {step}: val_loss={val_loss:.4f} (train {loss_val:.4f})", flush=True)
                    append_metric(metrics_path, {"event": "VAL", "step": step,
                                                 "val_loss": round(val_loss, 4),
                                                 "train_loss": round(loss_val, 4), "ts": time.time()})
                    if len(val_history) >= 4:
                        last4 = val_history[-4:]
                        if last4[0] < last4[1] < last4[2] < last4[3]:
                            watchdog_write("VAL_LOSS_DIVERGENCE", {"val_history": [round(v, 4) for v in last4]})
                            val_diverged = True
            if is_ddp:
                t = torch.tensor([1 if val_diverged else 0], device=device)
                torch.distributed.all_reduce(t)
                val_diverged = t.item() > 0
            if val_diverged:
                break

            if (step % save_steps == 0 or step == max_steps
                    or (time.time() - last_save_time) >= save_interval_sec):
                if is_main:
                    step_path = save_checkpoint(raw_model, optimizer, step, loss_val, val_loss,
                                                ckpt_dir, latest_path, resume_state=resume_state, scaler=scaler)
                    last_save_time = time.time()
                    print(f"--> [Checkpoint] saved step {step} -> {step_path}", flush=True)
                    append_metric(metrics_path, {"event": "CKPT", "step": step, "path": step_path, "ts": time.time()})

            if epoch_done:
                if is_main:
                    watchdog_write("EPOCH_COMPLETE_SINGLE_PASS")
                break

            stop_requested = False
            if is_main:
                stop_file = os.path.join(PROJECT_ROOT, "STOP_AND_SAVE")
                if os.path.exists(stop_file):
                    try:
                        os.remove(stop_file)
                    except OSError:
                        pass
                    stop_requested = True
            if is_ddp:
                t = torch.tensor([1 if stop_requested else 0], device=device)
                torch.distributed.all_reduce(t)
                stop_requested = t.item() > 0
            if stop_requested:
                if is_main:
                    watchdog_write("STOP_AND_SAVE_SIGNAL")
                break
      except Exception:
        # Never lose the diagnosis again: crash tracebacks go to metrics.json,
        # training.log AND the kernel log. FINAL below is marked unclean.
        crashed = True
        tb = traceback.format_exc()
        append_metric(metrics_path, {"event": "CRASH", "step": step,
                                     "error": tb[-1500:], "ts": time.time()})
        print(f"--> [CRASH] exception at step {step}:\n{tb}", flush=True)
        raise
    finally:
        _TELEMETRY_STOP.set()  # stop the live telemetry daemon before exit
        if _telemetry_thread is not None and _telemetry_thread.is_alive():
            _telemetry_thread.join(timeout=30)
        if step > start_step and is_main and not crashed:
            save_checkpoint(raw_model, optimizer, step, loss_val, val_loss, ckpt_dir, latest_path, resume_state=resume_state, scaler=scaler)
            print(f"--> [Final] checkpoint saved at step {step} (clean={not crashed})", flush=True)
            append_metric(metrics_path, {"event": "FINAL", "step": step,
                                         "clean": not crashed, "ts": time.time()})
        if is_ddp:
            import torch.distributed as dist
            dist.barrier()
            dist.destroy_process_group()

    elapsed = time.time() - start_time
    print(f"--> [DONE] step {step} | {tokens_seen:,} tokens in {elapsed/60:.1f} min "
          f"({tokens_seen/max(1.0,elapsed):.0f} tok/s)")
    print("SUCCESS: run #2 session ended cleanly.")

if __name__ == "__main__":
    train()
