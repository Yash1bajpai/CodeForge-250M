"""CPU-only incremental manifests, atomic handoff and actual trainer regressions."""
from contextlib import closing, nullcontext
import copy
import io
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from tokenizers import Tokenizer, models

from data.prepare_production import WindowWriter, document_key, tokenizer_identity
from models.architecture import CodeForgeModel
from training.dataset_extension import (commit_dataset_extension, dataset_fingerprint,
                                        extend_dataset_resume, extension_order, extension_position)
from training.production_data import LazyShardDataset, select_production_shards
from training.resume_safety import file_sha256, migrated_permutation, validate_resume_metadata
from training.token_accounting import budgeted_prefix
from training.train import save_checkpoint
from training.utils import get_lr_cosine_schedule


def _json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class DatasetExtensionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        python_rng, torch_rng = random.getstate(), torch.get_rng_state()
        self.addCleanup(random.setstate, python_rng)
        self.addCleanup(torch.set_rng_state, torch_rng)
        self.cfg = {
            "model": {"vocab_size": 128, "hidden_size": 16, "intermediate_size": 32,
                      "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
                      "max_position_embeddings": 8, "rms_norm_eps": 1e-5, "rope_theta": 10000,
                      "tie_word_embeddings": False},
            "training": {"batch_size_per_device": 1, "gradient_accumulation_steps": 1,
                         "learning_rate": .0006, "min_learning_rate": .00006,
                         "warmup_steps": 500, "max_steps": 16221, "val_steps": 1,
                         "val_batches": 1, "weight_decay": .1},
            "data": {"target_tokens": 1000}, "run": {"seed": 42},
            "checkpointing": {"output_dir": str(self.root / "checkpoints"), "push_to_hub": False}}
        self.context = {"config": {key: self.cfg.get(key) for key in ("model", "training", "data", "run")},
                        "stage": "foundation", "input_token_budget": 1000}
        self.parent = self.preparation("parent", 239, [(10, 8), (11, 8), (12, 3)], [(90, 4)])
        self.new = self.preparation("new", 240, [(20, 8), (21, 3)], [(95, 4)], self.parent)
        self.tokenizer = self.parent / "tokenizer"
        self.old_train = self.dataset(self.parent)
        self.val = self.dataset(self.parent, "val")
        records = {"train": self.old_train.audit_records, "validation": self.val.audit_records}
        fingerprint = dataset_fingerprint(records, self.tokenizer / "tokenizer.json", self.context)
        r = {"schema_version": 2, "fingerprint": fingerprint, "seqs_per_step": 1,
             "sample_cursor": 0, "input_tokens_seen": 0, "completed_updates": 0,
             "optimizer_step_count": 0, "order_version": "legacy-filtered-v1"}
        model = CodeForgeModel(self.cfg["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0006, betas=(.9, .95), weight_decay=.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=128)
        order, _ = migrated_permutation(self.old_train.rows)
        for step, index in enumerate(order, 1):
            self.update(model, optimizer, scaler, self.old_train, index, step)
            r["sample_cursor"] += 1
            r["input_tokens_seen"] += self.old_train.input_tokens(index)
            r["completed_updates"] += 1
            r["optimizer_step_count"] += 1
        (self.root / "checkpoints").mkdir()
        self.latest = self.root / "checkpoints/latest_checkpoint.pt"
        save_checkpoint(model, optimizer, step, 1., None, str(self.latest.parent), str(self.latest),
                        resume_state=r, scaler=scaler)
        self.checkpoint = torch.load(self.latest, weights_only=True)

    def preparation(self, name, number, train, val, parent=None):
        root = self.root / name
        (root / "tokenizer").mkdir(parents=True)
        specials = ["<|endoftext|>", "<|unk|>", "<|pad|>", "<|fim_prefix|>", "<|fim_middle|>",
                    "<|fim_suffix|>", "<|tool_call|>", "<|tool_result|>", "<|thinking|>",
                    "<|json_start|>", "<|json_end|>"]
        tok = Tokenizer(models.WordLevel({s: i for i, s in enumerate(specials + [f"token{i}" for i in range(11, 128)])},
                                        unk_token="<|unk|>"))
        tok.add_special_tokens(specials)
        tok.save(str(root / "tokenizer/tokenizer.json"))
        identity = tokenizer_identity(root / "tokenizer")
        _json(root / "tokenizer-identity.json", identity)
        writer = WindowWriter(root, number, {"seq_length": 8, "shard_sequences": 2}, identity["sha256"])
        digests = []
        if parent:
            with closing(sqlite3.connect((parent / "state.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
                digests = [row[0] for row in db.execute("SELECT digest FROM documents")]
        for split, values in (("train", train), ("validation", val)):
            for value, length in values:
                text = f"Synthetic document value {value} length {length}"
                digest = document_key(text)
                writer.add(("foundation", "fixture", split), [value] * length, [1] * length,
                           {"text": text, "stage": "foundation", "source": "fixture", "split": split,
                            "normalized_sha256": digest})
                digests.append(digest)
        writer.finish()
        with closing(sqlite3.connect(root / "state.sqlite3")) as db:
            db.execute("CREATE TABLE documents (digest TEXT PRIMARY KEY)")
            db.executemany("INSERT OR IGNORE INTO documents VALUES (?)", ((key,) for key in digests))
            db.commit()
        manifest = {"format_version": 2, "run_id": "fixture", "config_sha256": "c" * 64,
                    "pipeline_identity": {"sha256": "d" * 64}, "tokenizer": identity,
                    "state_sha256": file_sha256(root / "state.sqlite3"), "last_global_window": number,
                    "owned_windows": [{"number": number, "manifest_sha256": file_sha256(writer.destination / "manifest.json")}],
                    "parent": None}
        if parent:
            previous = json.loads((parent / "preparation-manifest.json").read_text())
            manifest["parent"] = {"manifest_sha256": file_sha256(parent / "preparation-manifest.json"),
                                  "last_global_window": previous["last_global_window"],
                                  "previous_run_id": previous["run_id"]}
        _json(root / "preparation-manifest.json", manifest)
        return root

    def dataset(self, root, split="train"):
        descriptors = select_production_shards(root, "foundation", split, root / "tokenizer", 8, 128)
        return LazyShardDataset(root, seq_length=8, vocab_size=128, descriptors=descriptors,
                                return_input_tokens=split == "train")

    def extend(self, checkpoint=None, parent=None, new=None, val=None, context=None, seed=101):
        checkpoint = self.checkpoint if checkpoint is None else checkpoint
        new = self.new if new is None else new
        return extend_dataset_resume(checkpoint, checkpoint["resume_state"]["fingerprint"],
                                     self.parent if parent is None else parent, new, self.dataset(new),
                                     self.val if val is None else val, self.tokenizer,
                                     self.context if context is None else context, seed)

    def update(self, model, optimizer, scaler, dataset, index, step):
        x, y, _ = dataset[index]
        lr = get_lr_cosine_schedule(step, 500, 16221, .0006, .00006)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        _, loss = model(x.unsqueeze(0), y.unsqueeze(0))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

    def continue_cpu(self, checkpoint, root, limit=None):
        # Optimizer.load_state_dict can alias same-device tensors; emulate a fresh load.
        checkpoint = copy.deepcopy(checkpoint)
        dataset = self.dataset(root)
        r = copy.deepcopy(checkpoint["resume_state"])
        records = {"train": dataset.audit_records, "validation": self.val.audit_records}
        fingerprint = dataset_fingerprint(records, self.tokenizer / "tokenizer.json", self.context, r)
        validate_resume_metadata(checkpoint, fingerprint, 1, 1000)
        cursor, tokens = extension_position(dataset, r)
        order = extension_order(dataset, r)
        self.assertEqual(r["input_tokens_seen"], tokens + sum(dataset.input_tokens(i) for i in order[:cursor]))
        model = CodeForgeModel(self.cfg["model"])
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler = torch.amp.GradScaler("cpu")
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        random.setstate(checkpoint["rng_state"]["python"])
        torch.set_rng_state(checkpoint["rng_state"]["torch_cpu"])
        remaining, _ = budgeted_prefix(dataset, order[cursor:], r["input_tokens_seen"], 1000)
        step = checkpoint["step"]
        for index in remaining[:limit]:
            step += 1
            self.update(model, optimizer, scaler, dataset, index, step)
            r["sample_cursor"] += 1
            r["input_tokens_seen"] += dataset.input_tokens(index)
            r["completed_updates"] += 1
            r["optimizer_step_count"] += 1
        save_checkpoint(model, optimizer, step, 1., None, str(self.latest.parent), str(self.latest),
                        keep=100, resume_state=r, scaler=scaler)
        return torch.load(self.latest, weights_only=True)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_activation_preserves_all_ml_state_and_rng_and_next_lr(self):
        python_rng, torch_rng = random.getstate(), torch.get_rng_state()
        r = self.extend()
        commit_dataset_extension(self.latest, self.checkpoint, r)
        saved = torch.load(self.latest, weights_only=True)
        for key in self.checkpoint.keys() - {"resume_state"}:
            self.assert_tree_equal(saved[key], self.checkpoint[key])
        self.assertEqual(random.getstate(), python_rng)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        for field in ("sample_cursor", "input_tokens_seen", "completed_updates", "optimizer_step_count"):
            self.assertEqual(r[field], self.checkpoint["resume_state"][field])
        result = self.continue_cpu(saved, self.new, 1)
        self.assertEqual(result["step"], self.checkpoint["step"] + 1)
        self.assertEqual(result["optimizer_state_dict"]["param_groups"][0]["lr"], .0006 * result["step"] / 500)
        self.assertEqual(result["scaler_state_dict"]["_growth_tracker"], saved["scaler_state_dict"]["_growth_tracker"] + 1)

    def test_incremental_order_excludes_old_rows_and_keeps_cumulative_tokens(self):
        r = self.extend()
        new = self.dataset(self.new)
        order = extension_order(new, r)
        self.assertEqual(sorted(order), list(range(len(new))))
        old_rows = {tuple(self.old_train[i][0].tolist()) for i in range(len(self.old_train))}
        new_rows = {tuple(new[i][0].tolist()) for i in order}
        self.assertFalse(old_rows & new_rows)
        result = self.continue_cpu({**self.checkpoint, "resume_state": r}, self.new)
        self.assertEqual(result["resume_state"]["sample_cursor"], len(self.old_train) + len(new))
        self.assertEqual(result["resume_state"]["input_tokens_seen"],
                         self.checkpoint["resume_state"]["input_tokens_seen"] + sum(new.input_tokens(i) for i in order))
        capped, limited = budgeted_prefix(new, order, r["input_tokens_seen"], r["input_tokens_seen"] + new.input_tokens(order[0]))
        self.assertEqual(capped, order[:1])
        self.assertTrue(limited)

    def test_parent_fingerprint_and_original_config_must_match(self):
        bad = copy.deepcopy(self.checkpoint)
        bad["resume_state"]["fingerprint"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.extend(checkpoint=bad)
        with self.assertRaisesRegex(ValueError, "Parent fingerprint mismatch"):
            extend_dataset_resume(self.checkpoint, "0" * 64, self.parent, self.new,
                                  self.dataset(self.new), self.val, self.tokenizer, self.context, 101)
        changed = copy.deepcopy(self.context)
        changed["config"]["training"]["warmup_steps"] = 400
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.extend(context=changed)

    def test_tokenizer_and_preparation_recipe_changes_refused(self):
        manifest = self.new / "preparation-manifest.json"
        original = json.loads(manifest.read_text())
        for field, value in (("tokenizer", {}), ("config_sha256", "0" * 64),
                             ("pipeline_identity", {"sha256": "0" * 64})):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed[field] = value
                _json(manifest, changed)
                with self.assertRaises(ValueError):
                    self.extend()
        _json(manifest, original)

    def test_consistently_replaced_new_tokenizer_is_refused(self):
        path = self.new / "tokenizer/tokenizer.json"
        tokenizer = json.loads(path.read_text())
        tokenizer["model"]["vocab"]["replacement"] = tokenizer["model"]["vocab"].pop("token20")
        _json(path, tokenizer)
        identity = tokenizer_identity(self.new / "tokenizer")
        _json(self.new / "tokenizer-identity.json", identity)
        window_path = self.new / "windows/window-000240/manifest.json"
        window = json.loads(window_path.read_text())
        window["tokenizer_sha256"] = identity["sha256"]
        _json(window_path, window)
        manifest_path = self.new / "preparation-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tokenizer"] = identity
        manifest["owned_windows"][0]["manifest_sha256"] = file_sha256(window_path)
        _json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "tokenizer changed"):
            self.extend()

    def test_old_document_and_validation_leak_refused(self):
        for name, value, length in (("repeat-train", 10, 8), ("leak-validation", 90, 4)):
            with self.subTest(name=name):
                duplicate = self.preparation(name, 240, [(value, length)], [(96, 4)], self.parent)
                with self.assertRaisesRegex(ValueError, "repeats an old/new document"):
                    self.extend(new=duplicate)

    def test_lost_dedup_history_wrong_parent_and_nonincremental_windows_refused(self):
        path = self.new / "preparation-manifest.json"
        original = json.loads(path.read_text())
        changed = copy.deepcopy(original)
        changed["parent"]["manifest_sha256"] = "0" * 64
        _json(path, changed)
        with self.assertRaisesRegex(ValueError, "directly descend"):
            self.extend()
        _json(path, original)
        wrong = self.preparation("wrong-range", 242, [(40, 8)], [(97, 4)], self.parent)
        with self.assertRaisesRegex(ValueError, "contiguous NEW windows"):
            self.extend(new=wrong)
        with closing(sqlite3.connect(self.new / "state.sqlite3")) as db:
            db.execute("DELETE FROM documents WHERE digest=?", (document_key("Synthetic document value 10 length 8"),))
            db.commit()
        original["state_sha256"] = file_sha256(self.new / "state.sqlite3")
        _json(path, original)
        with self.assertRaisesRegex(ValueError, "lost the parent's"):
            self.extend()

    def test_unconsumed_parent_or_corrupt_optimizer_or_counter_refused(self):
        for field, value in (("sample_cursor", 1), ("input_tokens_seen", 1),
                             ("optimizer_step_count", 99), ("completed_updates", 99), ("seqs_per_step", 9),
                             ("legacy_sample_cursor", 1)):
            with self.subTest(field=field):
                bad = copy.deepcopy(self.checkpoint)
                bad["resume_state"][field] = value
                with self.assertRaises(ValueError):
                    self.extend(checkpoint=bad)
        for field in ("rng_state", "scaler_state_dict"):
            with self.subTest(missing=field):
                bad = copy.deepcopy(self.checkpoint)
                del bad[field]
                with self.assertRaisesRegex(ValueError, "saved RNG and scaler history"):
                    self.extend(checkpoint=bad)

    def test_frozen_validation_and_extension_identity_cannot_change_on_resume(self):
        r = self.extend()
        records = {"train": self.dataset(self.new).audit_records, "validation": self.dataset(self.new, "val").audit_records}
        with self.assertRaisesRegex(ValueError, "validation membership"):
            dataset_fingerprint(records, self.tokenizer / "tokenizer.json", self.context, r)
        records["validation"] = self.val.audit_records
        self.assertEqual(dataset_fingerprint(records, self.tokenizer / "tokenizer.json", self.context, r), r["fingerprint"])
        changed = copy.deepcopy(r)
        changed["order_seed"] += 1
        with self.assertRaisesRegex(ValueError, "lineage/order"):
            dataset_fingerprint(records, self.tokenizer / "tokenizer.json", self.context, changed)
        changed_context = copy.deepcopy(self.context)
        changed_context["input_token_budget"] += 1
        with self.assertRaisesRegex(ValueError, "cumulative budget changed"):
            dataset_fingerprint(records, self.tokenizer / "tokenizer.json", changed_context, r)

    def test_repeated_extension_refused_and_two_successive_extensions_work(self):
        r = self.extend()
        first = {**self.checkpoint, "resume_state": r}
        with self.assertRaisesRegex(ValueError, "already extended.*plain --resume"):
            extend_dataset_resume(first, self.checkpoint["resume_state"]["fingerprint"], self.parent, self.new,
                                  self.dataset(self.new), self.val, self.tokenizer, self.context, 101)
        completed = self.continue_cpu(first, self.new)
        third = self.preparation("third", 241, [(40, 8), (41, 3)], [(97, 4)], self.new)
        next_state = self.extend(checkpoint=completed, parent=self.new, new=third, seed=202)
        self.assertEqual(len(next_state["lineage_chain"]), 2)
        self.assertEqual(next_state["parent_fingerprint"], r["fingerprint"])
        result = self.continue_cpu({**completed, "resume_state": next_state}, third)
        self.assertEqual(result["resume_state"]["sample_cursor"], len(self.old_train) + len(self.dataset(self.new)) + len(self.dataset(third)))

    def test_interrupted_activation_is_atomic_and_interrupted_training_matches(self):
        r = self.extend()
        old_hash = file_sha256(self.latest)
        with patch("training.dataset_extension.os.replace", side_effect=OSError("interrupted rename")):
            with self.assertRaises(OSError):
                commit_dataset_extension(self.latest, self.checkpoint, r)
        self.assertEqual(file_sha256(self.latest), old_hash)
        commit_dataset_extension(self.latest, self.checkpoint, self.extend())
        activated = torch.load(self.latest, weights_only=True)
        uninterrupted = self.continue_cpu(activated, self.new)
        interrupted = self.continue_cpu(activated, self.new, 1)
        resumed = self.continue_cpu(interrupted, self.new)
        self.assertEqual(len(resumed["resume_state"]["lineage_chain"]), 1)
        self.assert_tree_equal(uninterrupted, resumed)

    def test_foundation_only_and_plain_resume_remains_strict(self):
        for stage in ("instruction", "code_instruction", "tool_instruction"):
            context = {**self.context, "stage": stage}
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "foundation only"):
                self.extend(context=context)
        self.assertEqual(validate_resume_metadata(self.checkpoint, self.checkpoint["resume_state"]["fingerprint"], 1, 1000),
                         self.checkpoint["resume_state"])
        records = {"train": self.dataset(self.new).audit_records, "validation": self.val.audit_records}
        changed = dataset_fingerprint(records, self.tokenizer / "tokenizer.json", self.context)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            validate_resume_metadata(self.checkpoint, changed, 1, 1000)

    def test_unclosed_or_unverified_database_refused(self):
        journal = self.new / "state.sqlite3-journal"
        journal.write_bytes(b"unclosed producer")
        with self.assertRaisesRegex(ValueError, "closed preparation database"):
            self.extend()
        journal.unlink()
        path = self.new / "preparation-manifest.json"
        manifest = json.loads(path.read_text())
        manifest["state_sha256"] = None
        _json(path, manifest)
        with self.assertRaisesRegex(ValueError, "hash-verified final preparation"):
            self.extend()

    def test_extension_flags_require_explicit_resume_and_cannot_reset_state(self):
        original = file_sha256(self.latest)
        flags = ["--resume", "--extend-dataset", "--parent-fingerprint", self.checkpoint["resume_state"]["fingerprint"],
                 "--parent-production-root", str(self.parent), "--new-data-manifest", str(self.new / "preparation-manifest.json"),
                 "--validation-root", str(self.parent), "--extension-seed", "101"]
        for rejected in (["--extend-dataset"], ["--parent-fingerprint", original],
                         flags + ["--init_from", str(self.latest)], flags + ["--from_scratch"]):
            with self.subTest(flags=rejected), self.assertRaises(SystemExit) as error:
                self.run_trainer_cpu(rejected)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(file_sha256(self.latest), original)

    def test_actual_cpu_trainer_cumulative_budget_never_overshoots_a_row(self):
        # Construct a parent whose ORIGINAL identity already bound this small cap.
        self.cfg["data"]["target_tokens"] = 24
        self.context["input_token_budget"] = 24
        records = {"train": self.old_train.audit_records, "validation": self.val.audit_records}
        parent = copy.deepcopy(self.checkpoint)
        parent["resume_state"]["fingerprint"] = dataset_fingerprint(records, self.tokenizer / "tokenizer.json", self.context)
        torch.save(parent, self.latest)
        log = self.run_trainer_cpu(["--resume", "--extend-dataset", "--parent-fingerprint", parent["resume_state"]["fingerprint"],
                                   "--parent-production-root", str(self.parent), "--new-data-manifest", str(self.new / "preparation-manifest.json"),
                                   "--validation-root", str(self.parent), "--extension-seed", "101"])
        self.assertIn("REAL_INPUT_TOKEN_BUDGET", log)
        saved = torch.load(self.latest, weights_only=True)
        dataset = self.dataset(self.new)
        first = extension_order(dataset, saved["resume_state"])[0]
        self.assertEqual(saved["resume_state"]["input_tokens_seen"], parent["resume_state"]["input_tokens_seen"] + dataset.input_tokens(first))
        self.assertLessEqual(saved["resume_state"]["input_tokens_seen"], 24)
        self.assertEqual(saved["resume_state"]["sample_cursor"], parent["resume_state"]["sample_cursor"] + 1)

    def run_trainer_cpu(self, flags, cfg=None):
        """Override ONLY CUDA hardware gating/autocast; execute the real trainer on CPU."""
        import training.train as engine
        cfg = self.cfg if cfg is None else cfg
        config = self.root / "trainer.yaml"
        config.write_text(yaml.safe_dump(cfg))
        output = io.StringIO()
        available = iter((False, True))  # choose CPU device, then pass the legacy GPU-only gate
        scaler_type = torch.amp.GradScaler
        with patch.object(engine, "PROJECT_ROOT", str(self.root)), \
                patch.object(engine, "Logger", return_value=output), \
                patch.object(sys, "argv", ["train", "--config", str(config), *flags]), \
                patch.object(sys, "stdout", output), patch.object(sys, "stderr", output), \
                patch.dict(os.environ, {"CF_TELEMETRY_DISABLED": "1", "CUDA_VISIBLE_DEVICES": "-1",
                                        "WORLD_SIZE": "1", "CF_VAL_DIR": "", "CF_ABSOLUTE_STOP_EPOCH": "inf"}), \
                patch("torch.cuda.is_available", side_effect=lambda: next(available, False)), \
                patch("torch.cuda.get_device_capability", return_value=(7, 5)), \
                patch("torch.cuda.get_device_name", return_value="CPU test; no CUDA"), \
                patch("torch.amp.GradScaler", side_effect=lambda *a, **kw: scaler_type("cpu")), \
                patch("torch.amp.autocast", side_effect=lambda *a, **kw: nullcontext()):
            engine.train()
        return output.getvalue()

    def test_actual_cpu_trainer_extension_plain_resume_and_weights_only_start(self):
        parent_hash = file_sha256(self.latest)
        ordinary = self.run_trainer_cpu(["--resume", "--production-root", str(self.parent), "--session-steps", "1"])
        self.assertIn("SUCCESS", ordinary)
        self.assertEqual(file_sha256(self.latest), parent_hash)  # Exhausted means no repeated rows/update.
        log = self.run_trainer_cpu(["--resume", "--extend-dataset", "--parent-fingerprint", self.checkpoint["resume_state"]["fingerprint"],
                                   "--parent-production-root", str(self.parent), "--new-data-manifest", str(self.new / "preparation-manifest.json"),
                                   "--validation-root", str(self.parent), "--extension-seed", "101", "--session-steps", "1"])
        self.assertIn("SUCCESS", log)
        interrupted = torch.load(self.latest, weights_only=True)
        self.assertEqual(interrupted["step"], self.checkpoint["step"] + 1)
        self.assertEqual(len(interrupted["resume_state"]["lineage_chain"]), 1)
        log = self.run_trainer_cpu(["--resume", "--production-root", str(self.new), "--validation-root", str(self.parent), "--session-steps", "1"])
        self.assertIn("SUCCESS", log)
        completed = torch.load(self.latest, weights_only=True)
        self.assertEqual(completed["resume_state"]["sample_cursor"], len(self.old_train) + len(self.dataset(self.new)))
        expected = self.continue_cpu({**self.checkpoint, "resume_state": self.extend()}, self.new)
        for key in ("model_state_dict", "optimizer_state_dict", "scaler_state_dict", "resume_state", "rng_state"):
            self.assert_tree_equal(completed[key], expected[key])
        source = self.root / "warmstart-parent.pt"
        torch.save(self.checkpoint, source)
        source_hash = file_sha256(source)
        warm_cfg = copy.deepcopy(self.cfg)
        warm_cfg["checkpointing"]["output_dir"] = str(self.root / "warm-checkpoints")
        log = self.run_trainer_cpu(["--init_from", str(source), "--production-root", str(self.parent), "--session-steps", "1"], warm_cfg)
        warm = torch.load(self.root / "warm-checkpoints/latest_checkpoint.pt", weights_only=True)
        self.assertIn("weights only", log)
        self.assertEqual(warm["step"], 1)
        self.assertEqual(warm["resume_state"]["sample_cursor"], 1)
        self.assertEqual(warm["resume_state"]["optimizer_step_count"], 1)
        self.assertNotIn("extension", warm["resume_state"])
        self.assertEqual(file_sha256(source), source_hash)
        warm_model = CodeForgeModel(self.cfg["model"])
        warm_model.load_state_dict(self.checkpoint["model_state_dict"])
        warm_optimizer = torch.optim.AdamW(warm_model.parameters(), lr=.0006, betas=(.9, .95), weight_decay=.1)
        warm_scaler = torch.amp.GradScaler("cpu")
        first_index = migrated_permutation(self.old_train.rows)[0][0]
        self.update(warm_model, warm_optimizer, warm_scaler, self.old_train, first_index, 1)
        self.assert_tree_equal(warm["model_state_dict"], warm_model.state_dict())
        self.assertNotEqual(file_sha256(self.latest), parent_hash)


if __name__ == "__main__":
    unittest.main()
