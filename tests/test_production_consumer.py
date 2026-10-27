"""Tiny CPU fixtures for consuming the producer's committed manifest format."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from training.production_data import LazyShardDataset, select_production_shards
from training.resume_safety import manifest_fingerprint


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class ProductionConsumerTests(unittest.TestCase):
    def _fixture(self, specs=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        tokenizer = root / "tokenizer"
        tokenizer.mkdir()
        (tokenizer / "tokenizer.json").write_text('{"fixture": true}')
        (tokenizer / "special_tokens_map.json").write_text('{"eos_token": "eos"}')
        files = {p.name: _sha(p) for p in sorted(tokenizer.iterdir()) if p.is_file()}
        identity = {"files": files,
                    "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
        _json(root / "tokenizer-identity.json", identity)
        window = root / "windows" / "window-000001"
        window.mkdir(parents=True)
        (window / "documents.jsonl").write_text('{}\n')
        specs = specs if specs is not None else [
            ("foundation", "code", "train", [8, 3]),
            ("foundation", "code", "validation", [4]),
            ("instruction", "chat", "train", [5]),
        ]
        manifest = {"window": 1, "tokenizer_sha256": identity["sha256"],
                    "seq_len": 8, "shards": [], "documents": 1,
                    "documents_sha256": _sha(window / "documents.jsonl"),
                    "discarded_tokens": 0, "valid_tokens": 0,
                    "padding_tokens": 0, "loss_tokens": 0}
        for index, (stage, source, split, lengths) in enumerate(specs, 1):
            directory = Path(stage) / source / split
            (window / directory).mkdir(parents=True, exist_ok=True)
            ids = torch.tensor([list(range(1, length + 1)) + [0] * (8 - length)
                                for length in lengths], dtype=torch.uint16)
            masks = []
            for length in lengths:
                mask = [0] + [1] * (length - 1) + [0] * (8 - length)
                if stage == "instruction":
                    mask[1] = 0
                masks.append(mask)
            loss = torch.tensor(masks, dtype=torch.uint8)
            payloads = {"file": ids,
                        "valid_file": {"valid_len": torch.tensor(lengths, dtype=torch.int32)},
                        "loss_file": loss}
            names = {"file": f"shard_{index:06d}.pt", "valid_file": f"valid_{index:06d}.pt",
                     "loss_file": f"loss_{index:06d}.pt"}
            item = {"stage": stage, "source": source, "split": split, "rows": len(lengths),
                    "valid_tokens": sum(lengths), "padding_tokens": 8 * len(lengths) - sum(lengths),
                    "loss_tokens": int(loss.sum())}
            for field, payload in payloads.items():
                path = directory / names[field]
                torch.save(payload, window / path)
                item[field] = path.as_posix()
                item[field + "_sha256"] = _sha(window / path)
            manifest["shards"].append(item)
            for field in ("valid_tokens", "padding_tokens", "loss_tokens"):
                manifest[field] += item[field]
        _json(window / "manifest.json", manifest)
        _json(root / "preparation-manifest.json", {
            "format_version": 2, "tokenizer": identity,
            "owned_windows": [{"number": 1, "manifest_sha256": _sha(window / "manifest.json")}],
            "parent": {"path": str(root / "must-not-follow")},
        })
        return root, tokenizer, window

    def _select(self, root, tokenizer, split="train", **kwargs):
        options = {"stage": "foundation", "split": split, "seq_length": 8, "vocab_size": 64}
        options.update(kwargs)
        return select_production_shards(root, tokenizer_dir=tokenizer, **options)

    def _manifest(self, window):
        return json.loads((window / "manifest.json").read_text())

    def _commit(self, root, window, manifest, pin=True):
        _json(window / "manifest.json", manifest)
        if pin:
            path = root / "preparation-manifest.json"
            preparation = json.loads(path.read_text())
            preparation["owned_windows"][0]["manifest_sha256"] = _sha(window / "manifest.json")
            _json(path, preparation)

    def _payload(self, root, window, field, value, index=0):
        manifest = self._manifest(window)
        item = manifest["shards"][index]
        path = window / item[field]
        torch.save(value, path)
        item[field + "_sha256"] = _sha(path)
        self._commit(root, window, manifest)

    def test_preparation_receipt_and_recipe_identity_are_bound_to_resume(self):
        root, tokenizer, _ = self._fixture()
        selected = self._select(root, tokenizer)
        before = LazyShardDataset(root, seq_length=8, vocab_size=64, descriptors=selected)
        path = root / "preparation-manifest.json"
        preparation = json.loads(path.read_text())
        preparation["config_sha256"] = "a" * 64
        _json(path, preparation)
        after = LazyShardDataset(root, seq_length=8, vocab_size=64,
                                 descriptors=self._select(root, tokenizer))
        self.assertNotEqual(
            manifest_fingerprint(before.audit_records, tokenizer / "tokenizer.json"),
            manifest_fingerprint(after.audit_records, tokenizer / "tokenizer.json"))

    def test_unsupported_preparation_manifest_version_is_refused(self):
        root, tokenizer, _ = self._fixture()
        path = root / "preparation-manifest.json"
        preparation = json.loads(path.read_text())
        preparation["format_version"] = 3
        _json(path, preparation)
        with self.assertRaisesRegex(ValueError, "manifest format"):
            self._select(root, tokenizer)

    def test_selects_only_foundation_requested_split_and_preserves_identity(self):
        root, tokenizer, window = self._fixture()
        train = self._select(root, tokenizer)
        validation = self._select(root, tokenizer, "val")
        self.assertEqual(len(train), 1)
        self.assertEqual(len(validation), 1)
        self.assertEqual(train[0]["identity"]["stage"], "foundation")
        self.assertEqual(train[0]["identity"]["split"], "train")
        self.assertEqual(validation[0]["identity"]["split"], "validation")
        self.assertNotEqual(train[0]["path"], validation[0]["path"])
        self.assertEqual(validation, self._select(root, tokenizer, "validation"))
        descriptor = train[0]
        self.assertTrue(Path(descriptor["path"]).is_absolute())
        self.assertEqual(descriptor["identity"]["window"], "windows/window-000001")
        self.assertEqual(descriptor["identity"]["manifest_sha256"], _sha(window / "manifest.json"))
        for field, descriptor_field in (("file", "path"), ("valid_file", "valid_path"),
                                        ("loss_file", "loss_path")):
            self.assertEqual(descriptor["identity"][field + "_sha256"], _sha(Path(descriptor[descriptor_field])))
        json.dumps(descriptor["identity"])

    def test_nested_descriptors_feed_lazy_dataset_without_split_leakage(self):
        root, tokenizer, _ = self._fixture()
        descriptors = self._select(root, tokenizer)
        dataset = LazyShardDataset(root, seq_length=8, vocab_size=64, descriptors=descriptors)
        self.assertEqual(len(dataset), 2)
        x, y = dataset[1]
        self.assertEqual(x.tolist(), [1, 2, 3, 0, 0, 0, 0])
        self.assertEqual(y.tolist(), [2, 3, -100, -100, -100, -100, -100])

    def test_single_window_needs_no_parent_receipt_or_discovery(self):
        root, tokenizer, window = self._fixture()
        (root / "tokenizer-identity.json").unlink()
        (root / "preparation-manifest.json").unlink()
        descriptors = self._select(window, tokenizer)
        self.assertEqual(len(descriptors), 1)
        self.assertEqual(descriptors[0]["identity"]["window"], ".")

    def test_does_not_glob_unowned_windows_or_follow_parent(self):
        root, tokenizer, _ = self._fixture()
        unowned = root / "windows" / "window-000002"
        unowned.mkdir()
        (unowned / "manifest.json").write_text("not json")
        (root / "must-not-follow").mkdir()
        (root / "must-not-follow" / "manifest.json").write_text("not json")
        self.assertEqual(len(self._select(root, tokenizer)), 1)

    def test_multiple_pinned_windows_keep_same_basename_identities_distinct(self):
        root, tokenizer, window = self._fixture()
        second = window.with_name("window-000002")
        shutil.copytree(window, second)
        manifest = self._manifest(second)
        manifest["window"] = 2
        _json(second / "manifest.json", manifest)
        path = root / "preparation-manifest.json"
        preparation = json.loads(path.read_text())
        preparation["owned_windows"].append({
            "number": 2, "manifest_sha256": _sha(second / "manifest.json")})
        _json(path, preparation)
        selected = self._select(root, tokenizer)
        self.assertEqual(len(selected), 2)
        self.assertEqual(Path(selected[0]["path"]).name, Path(selected[1]["path"]).name)
        self.assertNotEqual(selected[0]["path"], selected[1]["path"])
        self.assertNotEqual(selected[0]["identity"], selected[1]["identity"])
        dataset = LazyShardDataset(root, seq_length=8, vocab_size=64, descriptors=selected)
        self.assertEqual(len(dataset), 4)
        self.assertNotEqual(dataset.audit_records[0]["identity"], dataset.audit_records[1]["identity"])

    def test_missing_owned_manifest_does_not_fall_back_to_other_windows(self):
        root, tokenizer, window = self._fixture()
        (window / "manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing/unreadable"):
            self._select(root, tokenizer)

    def test_documents_provenance_checksum_is_verified(self):
        root, tokenizer, window = self._fixture()
        (window / "documents.jsonl").write_text("changed provenance\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self._select(root, tokenizer)

    def test_root_without_committed_manifest_does_not_glob(self):
        root, tokenizer, _ = self._fixture()
        (root / "preparation-manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "committed window manifest"):
            self._select(root, tokenizer)

    def test_tokenizer_identity_covers_all_files_not_only_tokenizer_json(self):
        root, tokenizer, _ = self._fixture()
        (tokenizer / "special_tokens_map.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            self._select(root, tokenizer)

    def test_extra_tokenizer_file_changes_aggregate_identity(self):
        root, tokenizer, _ = self._fixture()
        (tokenizer / "new-receipt.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            self._select(root, tokenizer)

    def test_missing_or_changed_tokenizer_receipt_fails_closed(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                root, tokenizer, _ = self._fixture()
                receipt = root / "tokenizer-identity.json"
                if missing:
                    receipt.unlink()
                else:
                    _json(receipt, {"files": {}, "sha256": "0" * 64})
                with self.assertRaises(ValueError):
                    self._select(root, tokenizer)

    def test_window_tokenizer_and_sequence_mismatch_are_rejected(self):
        for field, value in (("tokenizer_sha256", "0" * 64), ("seq_len", 9)):
            with self.subTest(field=field):
                root, tokenizer, window = self._fixture()
                manifest = self._manifest(window)
                manifest[field] = value
                self._commit(root, window, manifest)
                with self.assertRaises(ValueError):
                    self._select(root, tokenizer)

    def test_manifest_pin_prevents_silent_window_rewrite(self):
        root, tokenizer, window = self._fixture()
        path = window / "manifest.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self._select(root, tokenizer)

    def test_owned_window_must_have_a_manifest_pin(self):
        root, tokenizer, _ = self._fixture()
        path = root / "preparation-manifest.json"
        preparation = json.loads(path.read_text())
        del preparation["owned_windows"][0]["manifest_sha256"]
        _json(path, preparation)
        with self.assertRaisesRegex(ValueError, "manifest pin"):
            self._select(root, tokenizer)

    def test_missing_and_corrupt_sidecars_are_rejected(self):
        for field in ("valid_file", "loss_file"):
            for corrupt in (False, True):
                with self.subTest(field=field, corrupt=corrupt):
                    root, tokenizer, window = self._fixture()
                    item = self._manifest(window)["shards"][0]
                    path = window / item[field]
                    if corrupt:
                        path.write_bytes(path.read_bytes() + b"corrupt")
                    else:
                        path.unlink()
                    with self.assertRaises(ValueError):
                        self._select(root, tokenizer)

    def test_corruption_in_nonselected_split_is_not_ignored(self):
        root, tokenizer, window = self._fixture()
        validation = self._manifest(window)["shards"][1]
        (window / validation["loss_file"]).unlink()
        with self.assertRaises(ValueError):
            self._select(root, tokenizer, "train")

    def test_foundation_masks_must_match_all_and_only_valid_causal_targets(self):
        for position in (0, 1, 3):
            with self.subTest(position=position):
                root, tokenizer, window = self._fixture()
                item = self._manifest(window)["shards"][0]
                mask = torch.load(window / item["loss_file"], weights_only=True)
                mask[1, position] = 1 - mask[1, position]
                self._payload(root, window, "loss_file", mask)
                with self.assertRaisesRegex(ValueError, "loss mask|causal targets|padding"):
                    self._select(root, tokenizer)

    def test_sidecar_dtype_shape_and_valid_lengths_are_checked(self):
        bad_payloads = [
            ("valid_file", {"valid_len": torch.tensor([8, 3], dtype=torch.int64)}),
            ("valid_file", {"valid_len": torch.tensor([[8, 3]], dtype=torch.int32)}),
            ("valid_file", {"valid_len": torch.tensor([8, 1], dtype=torch.int32)}),
            ("valid_file", {"valid_len": torch.tensor([8, 9], dtype=torch.int32)}),
            ("valid_file", torch.tensor([8, 3], dtype=torch.int32)),
            ("loss_file", torch.zeros((2, 8), dtype=torch.int32)),
            ("loss_file", torch.zeros((2, 7), dtype=torch.uint8)),
            ("loss_file", torch.full((2, 8), 2, dtype=torch.uint8)),
            ("file", torch.zeros((2, 8), dtype=torch.int64)),
            ("file", torch.zeros((2, 7), dtype=torch.uint16)),
            ("file", torch.full((2, 8), 64, dtype=torch.uint16)),
        ]
        for field, payload in bad_payloads:
            with self.subTest(field=field, payload=str(payload)):
                root, tokenizer, window = self._fixture()
                self._payload(root, window, field, payload)
                with self.assertRaises(ValueError):
                    self._select(root, tokenizer)

    def test_shard_and_window_accounting_are_checked(self):
        for location in ("shard", "window"):
            for field in ("rows", "valid_tokens", "padding_tokens", "loss_tokens"):
                if location == "window" and field == "rows":
                    continue
                with self.subTest(location=location, field=field):
                    root, tokenizer, window = self._fixture()
                    manifest = self._manifest(window)
                    target = manifest["shards"][0] if location == "shard" else manifest
                    target[field] += 1
                    self._commit(root, window, manifest)
                    with self.assertRaises(ValueError):
                        self._select(root, tokenizer)

    def test_unsafe_relative_paths_are_rejected(self):
        for field in ("file", "valid_file", "loss_file"):
            for path in ("../outside.pt", "/tmp/outside.pt", "foundation/../../outside.pt",
                         "foundation\\outside.pt", "./outside.pt"):
                with self.subTest(field=field, path=path):
                    root, tokenizer, window = self._fixture()
                    manifest = self._manifest(window)
                    manifest["shards"][0][field] = path
                    self._commit(root, window, manifest)
                    with self.assertRaisesRegex(ValueError, "relative path"):
                        self._select(root, tokenizer)

    def test_symlink_escape_is_rejected(self):
        root, tokenizer, window = self._fixture()
        outside = root / "outside.pt"
        outside.write_bytes(b"not a tensor")
        (window / "escape.pt").symlink_to(outside)
        manifest = self._manifest(window)
        manifest["shards"][0]["file"] = "escape.pt"
        self._commit(root, window, manifest)
        with self.assertRaisesRegex(ValueError, "escapes root"):
            self._select(root, tokenizer)

    def test_duplicate_payload_paths_and_owned_windows_are_rejected(self):
        root, tokenizer, window = self._fixture()
        manifest = self._manifest(window)
        item = manifest["shards"][0]
        item["valid_file"] = item["file"]
        item["valid_file_sha256"] = item["file_sha256"]
        self._commit(root, window, manifest)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self._select(root, tokenizer)
        root, tokenizer, _ = self._fixture()
        path = root / "preparation-manifest.json"
        preparation = json.loads(path.read_text())
        preparation["owned_windows"] *= 2
        _json(path, preparation)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self._select(root, tokenizer)

    def test_empty_requested_split_and_empty_committed_prefix_fail_loudly(self):
        root, tokenizer, _ = self._fixture(specs=[("foundation", "code", "train", [3])])
        with self.assertRaisesRegex(ValueError, "No production shards selected"):
            self._select(root, tokenizer, "val")
        path = root / "preparation-manifest.json"
        preparation = json.loads(path.read_text())
        preparation["owned_windows"] = []
        _json(path, preparation)
        with self.assertRaisesRegex(ValueError, "No production shards selected"):
            self._select(root, tokenizer)

    def test_instruction_refused_before_even_legacy_path_access(self):
        with self.assertRaisesRegex(ValueError, "Instruction.*foundation only"):
            select_production_shards("legacy/no-manifest", "instruction", "train",
                                     "missing/tokenizer", 8, 64)

    def test_empty_window_manifest_fails_loudly(self):
        root, tokenizer, window = self._fixture(specs=[])
        with self.assertRaisesRegex(ValueError, "No production shards selected"):
            self._select(window, tokenizer)

    def test_owned_window_number_must_match_manifest(self):
        root, tokenizer, window = self._fixture()
        manifest = self._manifest(window)
        manifest["window"] = 2
        self._commit(root, window, manifest)
        with self.assertRaisesRegex(ValueError, "window number"):
            self._select(root, tokenizer)

    def test_uncommitted_window_is_rejected(self):
        root, tokenizer, window = self._fixture()
        temporary_window = window.with_name(window.name + ".part")
        window.rename(temporary_window)
        with self.assertRaisesRegex(ValueError, "Uncommitted"):
            self._select(temporary_window, tokenizer)


if __name__ == "__main__":
    unittest.main()
