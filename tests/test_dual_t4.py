"""Two small CPU/gloo processes, no corpus, CodeForge instance, cloud or GPU."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import dual_t4


def worker(rank, root):
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (45, 50))
    import random
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Dataset
    from training.train import save_checkpoint
    from training.resume_safety import optimizer_step_counts, validate_resume_metadata
    from training.validation_history import ValidationHistory, ValidationGuardStopped
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized(), "CPU regression must not initialize CUDA"
    device = torch.device('cpu')
    dist.init_process_group('gloo', init_method='file://' + str(root / 'rendezvous'),
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
        def forward(self, x, y):
            output = self.linear(x.float().unsqueeze(-1)).squeeze(-1)
            valid = y != -100
            loss = ((output - y)[valid] ** 2).mean()
            return output, loss
    class Rows(Dataset):
        def __init__(self):
            self.seen = []
        def __len__(self):
            return 1025
        def __getitem__(self, index):
            index = int(index)
            self.seen.append(index)
            count = index % 3 + 1
            x = torch.tensor([(index % 8) / 8, 0.25, 0.5])
            y = x * 0.5 + 0.125
            y[count:] = -100
            return x, y, count
    def single(model, optimizer, scaler, indices):
        optimizer.zero_grad()
        loader = DataLoader(Rows(), batch_size=4, sampler=indices)
        total = 0.0
        for x, y, counts in loader:
            _, loss = model(x, y)
            scaler.scale(loss * x.shape[0] / len(indices)).backward()
            total += loss.item() * x.shape[0]
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        return total / len(indices)
    try:
        torch.manual_seed(42)
        raw = Tiny()
        optimizer = torch.optim.AdamW(raw.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler('cpu', init_scale=8, growth_interval=2)
        permutation = torch.randperm(1025, generator=torch.Generator().manual_seed(17)).tolist()
        for step in range(3):
            single(raw, optimizer, scaler, permutation[step * 256:(step + 1) * 256])
        reference = copy.deepcopy(raw)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001)
        ref_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        ref_scaler = torch.amp.GradScaler('cpu')
        ref_scaler.load_state_dict(scaler.state_dict())
        model = torch.nn.parallel.DistributedDataParallel(raw, gradient_as_bucket_view=True)
        guard = ValidationHistory(1)
        for step, value in enumerate((1.0, 1.1, 1.2), 1):
            assert not guard.record(step, value)
        cursor = 768
        tokens = sum(index % 3 + 1 for index in permutation[:cursor])
        seen = []
        deltas = []
        for step, expected in ((4, 256), (5, 1)):
            rows = Rows()
            loader = DataLoader(rows, batch_size=4, sampler=dual_t4.RankIndices(permutation[cursor:], 4, rank))
            optimizer.zero_grad()
            loss, consumed, real_tokens = dual_t4.backward(model, iter(loader), expected, 4, rank, 4, scaler, device, None)
            assert consumed == expected
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert dual_t4.amp_step(scaler, optimizer, norm, device)
            reference_loss = single(reference, ref_optimizer, ref_scaler, permutation[cursor:cursor + expected])
            assert abs(loss - reference_loss) < 1e-6, (loss, reference_loss)
            delta = max(float((a - b).abs().max()) for a, b in zip(raw.parameters(), reference.parameters()))
            assert delta < 1e-6, delta
            deltas.append(delta)
            cursor += consumed
            tokens += real_tokens
            seen.extend(rows.seen)
            value, fired = dual_t4.validation(guard, step, lambda: 1.15, device)
            assert not fired
            if step == 4:
                random.seed(100 + rank)
                torch.manual_seed(200 + rank)
                rng = dual_t4.checkpoint_rng(device)
                expected_python, expected_torch = random.random(), torch.rand(3)
                resume = {'schema_version': 2, 'fingerprint': '0' * 64, 'seqs_per_step': 256,
                          'sample_cursor': cursor, 'input_tokens_seen': tokens,
                          'completed_updates': 4, 'optimizer_step_count': optimizer_step_counts(optimizer.state_dict()),
                          'validation_guard': guard.snapshot()}
                def write():
                    with (root / 'writer-rank').open('x') as stream:
                        stream.write(str(rank))
                    return save_checkpoint(raw, optimizer, step, loss, value, str(root), str(root / 'latest.pt'),
                                           resume_state=resume, scaler=scaler, rng_state=rng)
                dual_t4.rank_zero(write, device)
                saved = torch.load(root / 'latest.pt', weights_only=True)
                state = validate_resume_metadata(saved, '0' * 64, 256)
                assert state['sample_cursor'] == 1024
                cursor, tokens = state['sample_cursor'], state['input_tokens_seen']
                raw.load_state_dict(saved['model_state_dict'])
                optimizer.load_state_dict(saved['optimizer_state_dict'])
                scaler.load_state_dict(saved['scaler_state_dict'])
                guard = ValidationHistory.restore(saved, 1)
                assert [point['step'] for point in guard.history] == [1, 2, 3, 4]
                dual_t4.restore_rng(saved['rng_state'], rank, device, True)
                assert random.random() == expected_python
                torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0, atol=0)
                assert (root / 'writer-rank').read_text() == '0'
        assert cursor == 1025
        assert tokens == sum(index % 3 + 1 for index in permutation)
        gathered = [None, None]
        dist.all_gather_object(gathered, seen)
        assert not set(gathered[0]) & set(gathered[1])
        assert set(gathered[0] + gathered[1]) == set(permutation[768:])
        assert len(gathered[0]) + len(gathered[1]) == 257
        # One rank's local overflow MUST skip both optimizers and reset both scaler trackers.
        before = [p.detach().clone() for p in raw.parameters()]
        optimizer.zero_grad()
        x, y, _ = next(iter(DataLoader(Rows(), batch_size=1)))
        scaler.scale(model(x, y)[1]).backward()
        scaler.unscale_(optimizer)
        if rank == 1:
            next(raw.parameters()).grad.fill_(float('inf'))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        old_scale = scaler.get_scale()
        assert not dual_t4.amp_step(scaler, optimizer, norm, device)
        assert scaler.get_scale() == old_scale / 2
        assert scaler.state_dict()['_growth_tracker'] == 0
        assert optimizer_step_counts(optimizer.state_dict()) == 5
        for actual, previous in zip(raw.parameters(), before):
            torch.testing.assert_close(actual, previous, rtol=0, atol=0)
        # Divergent scaler state is detected BEFORE either optimizer updates.
        optimizer.zero_grad()
        scaler.scale(model(x, y)[1]).backward()
        scaler.unscale_(optimizer)
        if rank == 1:
            changed = scaler.state_dict()
            changed['scale'] *= 2
            scaler.load_state_dict(changed)
        try:
            dual_t4.amp_step(scaler, optimizer, 1.0, device)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Mismatched loss scales accepted')
        assert optimizer_step_counts(optimizer.state_dict()) == 5
        # A strict rising window fires on BOTH ranks and stays stopped after persistence.
        stop_guard = ValidationHistory(1)
        for step in (1, 2, 3):
            stop_guard.record(step, float(step))
        value, fired = dual_t4.validation(stop_guard, 4, lambda: 4.0, device)
        assert fired and stop_guard.stop['step'] == 4
        def write_guard():
            path = root / 'guard.json'
            with path.open('x') as stream:
                json.dump({'step': 4, 'val_loss': value, 'resume_state': {'validation_guard': stop_guard.snapshot()}}, stream)
        dual_t4.rank_zero(write_guard, device)
        try:
            ValidationHistory.restore(json.loads((root / 'guard.json').read_text()), 1)
        except ValidationGuardStopped:
            pass
        else:
            raise AssertionError('Persisted guard stop cleared')
        equal_guard = ValidationHistory(1)
        for step in (1, 2, 3):
            equal_guard.record(step, float(step))
        decisions = []
        for step, reading in ((4, 3.0), (5, 3.1), (6, 3.2), (7, 3.3)):
            decisions.append(dual_t4.validation(equal_guard, step, lambda: reading, device)[1])
        assert decisions == [False, False, False, True]
        try:
            dual_t4.rank_zero(lambda: (_ for _ in ()).throw(OSError('synthetic write failure')), device)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Rank0 write failure was not propagated')
        (root / f'rank-{rank}.json').write_text(json.dumps({'rank': rank, 'world_size': 2,
            'cursor': cursor, 'unique_consumed_rows': len(seen), 'tokens': tokens,
            'max_parameter_delta': max(deltas), 'full_step_rows': 256, 'tail_step_rows': 1,
            'overflow_skipped_both': True, 'scaler_mismatch_refused': True, 'guard_stop_persisted': True,
            'guard_equal_then_rise': decisions, 'rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
    finally:
        dist.destroy_process_group()


class DualT4Test(unittest.TestCase):
    def test_shards_preserve_microbatches_budget_and_resume(self):
        from training.token_accounting import budgeted_prefix
        order = [(index * 7) % 1033 for index in range(1033)]
        cursor = 768
        class Counts:
            def input_tokens(self, index):
                return index % 3 + 1
        dataset = Counts()
        prior = sum(dataset.input_tokens(index) for index in order[:cursor])
        limit = prior + sum(dataset.input_tokens(index) for index in order[cursor:cursor + 257])
        selected, stopped = budgeted_prefix(dataset, order[cursor:], prior, limit)
        self.assertTrue(stopped)
        self.assertEqual(selected, order[768:1025])
        shards = [dual_t4.RankIndices(selected, 4, rank) for rank in range(2)]
        ranks = [list(shard) for shard in shards]
        self.assertEqual([len(shard) for shard in shards], [129, 128])
        self.assertFalse(set(ranks[0]) & set(ranks[1]))
        reconstructed = []
        for start in range(0, 129, 4):
            for rank in range(2):
                reconstructed.extend(ranks[rank][start:start + 4])
        self.assertEqual(reconstructed, selected)
        self.assertEqual(sum(dataset.input_tokens(index) for index in reconstructed) + prior, limit)
        resumed_cursor = cursor + 256
        tail = order[resumed_cursor:1025]
        self.assertEqual(list(dual_t4.RankIndices(tail, 4, 0)), [order[1024]])
        self.assertEqual(list(dual_t4.RankIndices(tail, 4, 1)), [])

    def test_exact_geometry_and_hardware(self):
        self.assertEqual(dual_t4.geometry(4, 64, 2, 0), 32)
        self.assertEqual(dual_t4.geometry(16, 16, 2, 1), 8)
        for args in ((4, 63, 2, 0), (4, 64, 1, 0), (4, 64, 2, 2), (8, 64, 2, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                dual_t4.geometry(*args)
        for names, world, rank, local in ((['Tesla T4'], 2, 0, 0), (['Tesla T4'] * 3, 2, 0, 0),
                (['Tesla T4', 'Tesla P100'], 2, 0, 0), (['Tesla T4'] * 2, 1, 0, 0),
                (['Tesla T4'] * 2, 2, 1, 0)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                dual_t4.require_hardware(names, world, rank, local)

    def test_actual_cli_refuses_without_training_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = dict(os.environ, WORLD_SIZE='2', RANK='0', LOCAL_RANK='0', OMP_NUM_THREADS='1')
            env.pop('CF_TRAINING_APPROVED', None)
            result = subprocess.run([sys.executable, '-m', 'training.train', '--dual-t4', '--from_scratch',
                '--dual-t4-output', str(root / 'out'), '--session-steps', '1', '--dual-t4-max-minutes', '1'],
                env=env, capture_output=True, text=True, timeout=35)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires its own approved cloud invocation', result.stderr)
            self.assertFalse((root / 'out').exists())

    def test_two_process_gloo_math_resume_guard_and_scaler(self):
        available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        self.assertGreaterEqual(available, 768 * 1024, 'Need768MiB available for two CPU-only torch workers; HOLD')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       GLOO_SOCKET_IFNAME='lo', CUDA_VISIBLE_DEVICES='-1')
            env.pop('CF_TRAINING_APPROVED', None)
            processes = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(rank), str(root)],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for rank in range(2)]
            deadline = time.monotonic() + 120
            try:
                for process in processes:
                    output, _ = process.communicate(timeout=max(1, deadline - time.monotonic()))
                    self.assertEqual(process.returncode, 0, output)
                reports = [json.loads((root / f'rank-{rank}.json').read_text()) for rank in range(2)]
                self.assertEqual([report['cursor'] for report in reports], [1025, 1025])
                print('GLOO_DDP_RECEIPT ' + json.dumps(reports, sort_keys=True), flush=True)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(int(sys.argv[2]), Path(sys.argv[3]))
    else:
        unittest.main()
