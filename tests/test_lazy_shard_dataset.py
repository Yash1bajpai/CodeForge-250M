"""CPU-only synthetic coverage for bounded-memory shard consumption."""
import json
import random
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import torch

from training.production_data import LazyShardDataset
from training.resume_safety import file_sha256


class LazyShardDatasetTests(unittest.TestCase):
    @staticmethod
    def _write_shard(root, number, lengths, seq_length=8):
        path = root / f"shard_{number:04d}.pt"
        rows = [[(number * 8 + row + col) % 64 for col in range(seq_length)]
                for row in range(len(lengths))]
        tensor = torch.tensor(rows, dtype=torch.uint16)
        torch.save(tensor, path)
        torch.save({"valid_len": torch.tensor(lengths, dtype=torch.int32)},
                   root / f"valid_{number:04d}.pt")
        return path, tensor

    def test_random_access_partial_shards_matches_legacy_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = []
            paths = []
            for number, lengths in enumerate(([8, 3, 2], [5], [7, 8]), 1):
                path, tensor = self._write_shard(root, number, lengths)
                paths.append(path)
                expected.extend((row.long(), length) for row, length in zip(tensor, lengths))
            dataset = LazyShardDataset(root, seq_length=8, vocab_size=64)
            self.assertEqual(dataset.rows, [3, 1, 2])
            self.assertEqual(dataset.offsets, [0, 3, 4, 6])
            self.assertEqual(len(dataset), 6)
            order = list(range(6)) * 3
            random.Random(42).shuffle(order)
            for index in order:
                sample = dataset[index]
                self.assertEqual(len(sample), 2)
                x, y = sample
                sequence, valid = expected[index]
                target = sequence[1:].clone()
                target[valid - 1:] = -100
                self.assertTrue(torch.equal(x, sequence[:-1]))
                self.assertTrue(torch.equal(y, target))
            self.assertTrue(torch.equal(dataset[-1][0], expected[-1][0][:-1]))
            for index in (-7, 6):
                with self.assertRaises(IndexError):
                    dataset[index]
                with self.assertRaises(IndexError):
                    dataset.input_tokens(index)

    def test_cache_releases_previous_shard_before_loading_next(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_shard(root, 1, [8, 4])
            self._write_shard(root, 2, [3])
            dataset = LazyShardDataset(root, seq_length=8, vocab_size=64)
            self.assertFalse(hasattr(dataset, "data"))
            self.assertFalse(any(isinstance(value, torch.Tensor) and value.ndim == 2
                                 for value in vars(dataset).values()))
            references = []
            load = torch.load

            def tracked_load(*args, **kwargs):
                self.assertTrue(all(reference() is None for reference in references),
                                "A previous token shard remains resident during replacement")
                tensor = load(*args, **kwargs)
                references.append(weakref.ref(tensor))
                return tensor

            with patch("training.production_data.torch.load", side_effect=tracked_load) as mocked:
                dataset[0]
                dataset[1]
                self.assertEqual(mocked.call_count, 1)
                dataset[2]
                dataset[0]
                self.assertEqual(mocked.call_count, 3)
            self.assertEqual(sum(reference() is not None for reference in references), 1)
            state = dataset.__getstate__()
            self.assertFalse(any(isinstance(value, torch.Tensor) and value.ndim == 2
                                 for value in state.values()))

    def test_metadata_input_counts_and_opt_in_triples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_shard(root, 1, [2, 5, 8])
            dataset = LazyShardDataset(root, seq_length=8, vocab_size=64,
                                       return_input_tokens=True)
            with patch("training.production_data.torch.load", side_effect=AssertionError("not metadata-only")):
                self.assertEqual([dataset.input_tokens(i) for i in range(3)], [2, 5, 7])
            x, y, count = dataset[0]
            self.assertEqual(count, 2)  # Last real token is still an input on padded rows.
            self.assertEqual(int((y != -100).sum()), 1)
            self.assertEqual(x.numel(), 7)
            self.assertEqual(dataset[1][2], 5)
            self.assertEqual(dataset[2][2], 7)

    def test_missing_legacy_lengths_means_full_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "custom.pt"
            tensor = torch.tensor([[1, 2, 3, 4]], dtype=torch.uint16)
            torch.save(tensor, path)
            dataset = LazyShardDataset(root, seq_length=4, shard_files=[path], vocab_size=64)
            x, y = dataset[0]
            self.assertEqual(x.tolist(), [1, 2, 3])
            self.assertEqual(y.tolist(), [2, 3, 4])
            self.assertEqual(dataset.input_tokens(0), 3)

    def test_invalid_valid_lengths_are_rejected(self):
        invalid = [
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([9], dtype=torch.int32),
            torch.tensor([3, 4], dtype=torch.int32),
            torch.tensor([[3]], dtype=torch.int32),
            torch.tensor([3.5], dtype=torch.float32),
            torch.tensor([True]),
            torch.tensor([3], dtype=torch.int64),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_shard(root, 1, [3])
            for lengths in invalid:
                with self.subTest(shape=tuple(lengths.shape), dtype=lengths.dtype,
                                  values=lengths.tolist()):
                    torch.save({"valid_len": lengths}, root / "valid_0001.pt")
                    with self.assertRaises(ValueError):
                        LazyShardDataset(root, seq_length=8, vocab_size=64)
            torch.save({"not_valid_len": torch.tensor([3])}, root / "valid_0001.pt")
            with self.assertRaises(ValueError):
                LazyShardDataset(root, seq_length=8, vocab_size=64)

    def test_optional_legacy_loss_must_be_full_causal_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_shard(root, 1, [4])
            mask = torch.tensor([[0, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.uint8)
            loss = root / "loss_0001.pt"
            torch.save(mask, loss)
            dataset = LazyShardDataset(root, seq_length=8, vocab_size=64)
            self.assertEqual(dataset[0][1].tolist()[3:], [-100] * 4)
            invalid = [mask.clone(), mask.clone(), mask.clone(), mask.bool(), mask.flatten()]
            invalid[0][0, 2] = 0  # SFT-like omission of a real causal target.
            invalid[1][0, 0] = 1
            invalid[2][0, 5] = 1
            for value in invalid:
                with self.subTest(mask=value.tolist()):
                    torch.save(value, loss)
                    with self.assertRaises(ValueError):
                        LazyShardDataset(root, seq_length=8, vocab_size=64)

    def test_descriptor_sidecars_and_identity_are_audited_without_absolute_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            path = nested / "tokens.pt"
            valid_path = nested / "lengths.pt"
            loss_path = nested / "targets.pt"
            torch.save(torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]], dtype=torch.uint16), path)
            torch.save({"valid_len": torch.tensor([3], dtype=torch.int32)}, valid_path)
            torch.save(torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.uint8), loss_path)
            identity = {"window": "windows/window-000001", "source": "synthetic",
                        "stage": "foundation", "split": "train", "file": "nested/tokens.pt",
                        "manifest_sha256": "a" * 64}
            descriptors = [{"path": str(path), "valid_path": str(valid_path),
                            "loss_path": str(loss_path), "identity": identity}]
            dataset = LazyShardDataset(root, seq_length=8, vocab_size=64, descriptors=descriptors)
            record = dataset.audit_records[0]
            self.assertEqual(record["identity"], identity)
            self.assertEqual(record["sidecars"]["valid"]["sha256"], file_sha256(valid_path))
            self.assertEqual(record["sidecars"]["loss"]["sha256"], file_sha256(loss_path))
            self.assertNotIn(str(root), json.dumps(dataset.audit_records))
            self.assertEqual(dataset[0][1].tolist(), [2, 3, -100, -100, -100, -100, -100])

    def test_duplicate_shards_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self._write_shard(root, 1, [8])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                LazyShardDataset(root, seq_length=8, shard_files=[path, path], vocab_size=64)


if __name__ == "__main__":
    unittest.main()
