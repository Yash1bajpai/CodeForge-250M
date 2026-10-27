"""Synthetic CPU regressions for immutable data identity and real-ID budgets."""
import copy
from pathlib import Path
import random
import os
import tempfile
import unittest
import subprocess
import sys

import torch
from torch.utils.data import DataLoader

from training.production_data import LazyShardDataset
from training.resume_safety import (audit_shards, manifest_fingerprint, migrated_permutation,
                                    validate_resume_metadata)
from training.token_accounting import budgeted_prefix, input_token_budget
from training.train import save_checkpoint


class ProductionTrainingStateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.tokenizer = self.root / 'tokenizer.json'
        self.tokenizer.write_text('{}')
        self.path = self.root / 'shard_0.pt'
        torch.save(torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0],
                                 [1, 2, 3, 4, 5, 6, 7, 8],
                                 [1, 2, 3, 4, 0, 0, 0, 0]], dtype=torch.uint16), self.path)
        self.valid = self.root / 'valid_0.pt'
        torch.save({'valid_len': torch.tensor([3, 8, 4], dtype=torch.int32)}, self.valid)

    def test_instruction_and_approval_gates_precede_runtime_side_effects(self):
        project = Path(__file__).resolve().parents[1]
        config = self.root / 'config.yaml'
        config.write_text('model: {}\ntraining:\n  require_explicit_approval: true\n')
        for extra, message in [(['--stage', 'instruction'], 'assistant-loss-mask'),
                               ([], 'Training is blocked')]:
            with self.subTest(extra=extra):
                env = {'PATH': os.defpath, 'PYTHONPATH': str(project),
                       'CUDA_VISIBLE_DEVICES': '-1'}
                if os.name == 'nt':  # Winsock and DLL loading need the OS variables
                    for name in ('PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP'):
                        if os.environ.get(name):
                            env[name] = os.environ[name]
                result = subprocess.run(
                    [sys.executable, '-m', 'training.train', '--config', str(config), *extra],
                    cwd=self.root, env=env,
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 1)
                self.assertIn(message, result.stderr)
                self.assertFalse((self.root / 'training.log').exists())
                self.assertFalse((self.root / 'checkpoints').exists())

    def dataset(self):
        return LazyShardDataset(self.root, seq_length=8, vocab_size=16, return_input_tokens=True)

    def fingerprint(self, context=None):
        return manifest_fingerprint({'train': audit_shards([self.path], 8, 16), 'validation': []},
                                    self.tokenizer, context=context)

    def test_counts_inputs_not_padding_or_supervised_targets(self):
        ds = self.dataset()
        x, y, counts = next(iter(DataLoader(ds, batch_size=3)))
        self.assertEqual(counts.tolist(), [3, 7, 4])
        self.assertEqual(int(counts.sum()), 14)
        self.assertEqual(x.numel(), 21)
        self.assertEqual((y != -100).sum(dim=1).tolist(), [2, 7, 3])

    def test_budget_uses_stage_allocation_and_rejects_invalid_caps(self):
        config = {'run': {'target_input_tokens': 100}, 'data': {'target_tokens': 14}}
        self.assertEqual(input_token_budget(config), 14)
        self.assertEqual(input_token_budget(config, 9), 9)
        self.assertIsNone(input_token_budget({}))
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                input_token_budget({}, value)

    def test_budget_prefix_never_skips_next_row_or_overshoots(self):
        ds = self.dataset()
        self.assertEqual(budgeted_prefix(ds, [0, 1, 2], 0, 9), ([0], True))
        self.assertEqual(budgeted_prefix(ds, [0, 1, 2], 0, 10), ([0, 1], True))
        self.assertEqual(budgeted_prefix(ds, [0, 1, 2], 0, 14), ([0, 1, 2], False))
        self.assertEqual(budgeted_prefix(ds, [1, 2], 3, 9), ([], True))
        self.assertEqual(budgeted_prefix(ds, [1], 9, 9), ([], True))
        with self.assertRaises(ValueError):
            budgeted_prefix(ds, [0], 10, 9)

    def test_checkpoint_resume_preserves_real_budget_rng_optimizer_and_cursor(self):
        ds = self.dataset()
        order, _ = migrated_permutation(ds.rows)
        counts = [ds.input_tokens(i) for i in order]
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scaler = torch.amp.GradScaler('cpu')
        fingerprint = self.fingerprint({'budget': 14})
        resume = {'schema_version': 2, 'fingerprint': fingerprint, 'seqs_per_step': 1,
                  'sample_cursor': 1, 'input_tokens_seen': counts[0], 'completed_updates': 1,
                  'optimizer_step_count': 1, 'order_version': 'legacy-filtered-v1'}
        latest = str(self.root / 'latest.pt')
        save_checkpoint(model, optimizer, 1, 1.0, None, str(self.root), latest,
                        resume_state=resume, scaler=scaler)
        expected_python, expected_torch = random.random(), torch.rand(3)
        saved = torch.load(latest, weights_only=True)
        restored = validate_resume_metadata(saved, fingerprint, 1, 14)
        random.setstate(saved['rng_state']['python'])
        torch.set_rng_state(saved['rng_state']['torch_cpu'])
        self.assertEqual(random.random(), expected_python)
        self.assertTrue(torch.equal(torch.rand(3), expected_torch))
        restored_optimizer = torch.optim.AdamW(model.parameters())
        restored_optimizer.load_state_dict(saved['optimizer_state_dict'])
        self.assertEqual([int(v['step']) for v in restored_optimizer.state.values()], [1, 1])
        remaining, limited = budgeted_prefix(ds, order[restored['sample_cursor']:],
                                             restored['input_tokens_seen'], 14)
        self.assertFalse(limited)
        resumed_total = restored['input_tokens_seen'] + sum(ds.input_tokens(i) for i in remaining)
        self.assertEqual(resumed_total, 14)
        self.assertEqual(remaining, order[1:])
        self.assertTrue(saved['scaler_state_dict'])

    def test_identity_binds_length_mask_split_order_and_config(self):
        original = self.fingerprint({'training': {'max_steps': 2}})
        torch.save({'valid_len': torch.tensor([4, 8, 4], dtype=torch.int32)}, self.valid)
        self.assertNotEqual(original, self.fingerprint({'training': {'max_steps': 2}}))
        before_mask = self.fingerprint()
        loss_path = self.root / 'loss_0.pt'
        torch.save(torch.ones((3, 8), dtype=torch.uint8), loss_path)
        after_mask = self.fingerprint()
        self.assertNotEqual(before_mask, after_mask)
        mask = torch.ones((3, 8), dtype=torch.uint8)
        mask[0, 1] = 0
        torch.save(mask, loss_path)
        self.assertNotEqual(after_mask, self.fingerprint())
        records = audit_shards([self.path], 8, 16)
        train = manifest_fingerprint({'train': records, 'validation': []}, self.tokenizer)
        validation = manifest_fingerprint({'train': [], 'validation': records}, self.tokenizer)
        self.assertNotEqual(train, validation)
        self.assertNotEqual(self.fingerprint({'seed': 42}), self.fingerprint({'seed': 43}))
        self.assertNotEqual(manifest_fingerprint(records + [{'name': 'other'}], self.tokenizer),
                            manifest_fingerprint([{'name': 'other'}] + records, self.tokenizer))

    def test_resume_refuses_legacy_missing_invalid_counter_and_identity(self):
        state = {'optimizer_state_dict': {'state': {0: {'step': torch.tensor(1.)}}},
                 'resume_state': {'schema_version': 2, 'fingerprint': 'identity', 'seqs_per_step': 1,
                                  'sample_cursor': 1, 'input_tokens_seen': 3,
                                  'completed_updates': 1, 'optimizer_step_count': 1}}
        for field, value in [('schema_version', 1), ('input_tokens_seen', None),
                             ('input_tokens_seen', -1), ('input_tokens_seen', True),
                             ('input_tokens_seen', 11), ('sample_cursor', -1),
                             ('optimizer_step_count', 2), ('fingerprint', 'different'),
                             ('seqs_per_step', 2)]:
            changed = copy.deepcopy(state)
            changed['resume_state'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_resume_metadata(changed, 'identity', 1, 10)
        self.assertEqual(validate_resume_metadata(state, 'identity', 1, 10)['input_tokens_seen'], 3)


if __name__ == '__main__':
    unittest.main()
