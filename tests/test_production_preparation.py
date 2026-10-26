from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tokenizers import Tokenizer, models, pre_tokenizers

from data.instruction_format import normalize_instruction
from data.prepare_fresh import file_sha256
from data.train_tokenizer import train_custom_bpe_tokenizer
from data.prepare_production import (WindowWriter, document_key, document_split, encode_record,
                                    prepare, sample_tokenizer, source_rows, tokenizer_identity,
                                    verify_window)


def small_tokenizer():
    tok = Tokenizer(models.WordLevel({"<|unk|>": 0, "<|endoftext|>": 1, "Hello": 2}, unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(["<|endoftext|>"])
    return tok


def _preparation_fixture(root):
    first = root / "first"
    (first / "tokenizer").mkdir(parents=True)
    (first / "tokenizer-sample").mkdir()
    (first / "tokenizer-sample/foundation_dedup.jsonl").write_text("")
    small_tokenizer().save(str(first / "tokenizer/tokenizer.json"))
    (first / "tokenizer-identity.json").write_text(json.dumps(tokenizer_identity(first / "tokenizer")))
    config = {"run_id": "fixture", "purpose": "data-preparation-only", "seed": 42,
              "vocab_size": 3, "seq_length": 32, "validation_fraction": .01,
              "max_document_chars": 1000, "tokenizer_sample_chars": 0,
              "session_seconds": 600, "session_max_tokens": 1, "shard_sequences": 2,
              "checkpoint_documents": 2, "min_free_bytes": 0,
              "benchmark_exclusion": {"window_tokens": 32, "minimum_distinct_tokens": 12},
              "sources": [{"name": "fixture", "stage": "foundation", "text_field": "text",
                           "repo": "fixture/source", "revision": "a" * 40,
                           "target_tokens": 100000, "allowed_licenses": ["mit"]}]}
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    texts = ["def alpha(value):\n    return value + 111\n",
             "def beta(value):\n    return value + 222\n",
             "def alpha(value):\n    return value + 111\n",
             "def gamma(value):\n    return value + 333\n"]
    repo = next(f"repo-{i}" for i in range(100) if document_split(
        {"repo_name": f"repo-{i}"}, "", 42, .01) == "train")

    def rows(spec, cursor):
        for index in range(cursor["row"], len(texts)):
            cursor["row"] = index + 1
            yield {"text": texts[index], "repo_name": repo}, "fixture.jsonl", index, "mit"
        cursor["file"] = 1

    return first, config_path, config, rows


class _JsonFileRows:
    """A retained file iterator, so collection cannot substitute for ownership."""

    def __init__(self, path, close_failure=False):
        self.handle = path.open(encoding="utf-8")
        self.close_failure = close_failure

    def __iter__(self):
        return self

    def __next__(self):
        return json.loads(next(self.handle))

    def close(self):
        self.handle.close()
        if self.close_failure:
            raise OSError("Stream close failed")


class ProductionPreparationTests(unittest.TestCase):
    def test_instruction_offsets_exclude_user_and_supervise_answer_eos(self):
        record = normalize_instruction({"instruction": "A private request", "output": "Hello"}, "opencoder")
        tok = small_tokenizer()
        ids, mask = encode_record(tok, record, {}, document_key(record["text"]), 2048)
        offsets = tok.encode(record["text"], add_special_tokens=False).offsets
        supervised = [record["text"][start:end] for (start, end), flag in zip(offsets, mask) if flag]
        self.assertEqual(supervised, ["Hello", "<|endoftext|>"])
        self.assertEqual(len(ids), len(mask))

    def test_long_instruction_is_rejected_not_truncated(self):
        record = normalize_instruction({"instruction": "Explain", "output": "Hello " * 40}, "opencoder")
        self.assertEqual(encode_record(small_tokenizer(), record, {}, document_key(record["text"]), 16), (None, None))

    def test_repository_split_does_not_depend_on_file_contents(self):
        row = {"repo_name": "owner/project"}
        self.assertEqual(document_split(row, "first function", 42, .1),
                         document_split(row, "completely different contents", 42, .1))
        self.assertEqual(document_key("a   b\n c"), document_key("a b c"))

    def test_packed_tail_and_padding_have_explicit_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = WindowWriter(Path(directory), 1, {"seq_length": 8, "shard_sequences": 2}, "identity")
            writer.add(("foundation", "code", "train"), list(range(9)), [1] * 9, {"stage": "foundation"})
            manifest = writer.finish()
            verified = verify_window(writer.destination)
            self.assertEqual(verified["valid_tokens"], 8)
            self.assertEqual(verified["discarded_tokens"], 1)
            self.assertEqual(verified["loss_tokens"], 7)
            self.assertEqual(manifest["padding_tokens"], 0)

    def test_source_rows_releases_file_at_exhaustion_and_explicit_close(self):
        spec = {"name": "fixture", "text_field": "text", "allowed_licenses": ["mit"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"text": "first"}\n{"text": "second"}\n', encoding="utf-8")
            for exhaust in (False, True):
                with self.subTest(exhaust=exhaust):
                    file_rows = _JsonFileRows(path)
                    cursor = {"file": 0, "row": 0}
                    with patch("data.prepare_production.source_files", return_value=([path.name], "mit")), \
                            patch("data.prepare_production.stream_file", return_value=file_rows):
                        rows = source_rows(spec, cursor)
                        try:
                            next(rows)
                            self.assertFalse(file_rows.handle.closed)
                            if exhaust:
                                list(rows)
                            else:
                                rows.close()
                            with self.assertRaises(ValueError):
                                file_rows.handle.read()
                            self.assertEqual(cursor, {"file": 1, "row": 0} if exhaust
                                             else {"file": 0, "row": 1})
                        finally:
                            rows.close()
                            file_rows.handle.close()

    def test_source_schema_failure_remains_primary_when_stream_close_also_fails(self):
        spec = {"name": "fixture", "text_field": "text", "allowed_licenses": ["mit"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"wrong_field": "first"}\n', encoding="utf-8")
            file_rows = _JsonFileRows(path, close_failure=True)
            with patch("data.prepare_production.source_files", return_value=([path.name], "mit")), \
                    patch("data.prepare_production.stream_file", return_value=file_rows):
                try:
                    with self.assertRaisesRegex(ValueError, "configured text field"):
                        next(source_rows(spec, {"file": 0, "row": 0}))
                    with self.assertRaises(ValueError):
                        file_rows.handle.read()
                finally:
                    file_rows.handle.close()

    def test_tokenizer_sampling_failure_releases_source_and_sqlite_transaction(self):
        spec = {"name": "fixture", "text_field": "text", "allowed_licenses": ["mit"],
                "tokenizer_chars": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "rows.jsonl"
            path.write_text('{"text": "first"}\n', encoding="utf-8")
            file_rows = _JsonFileRows(path)
            with patch("data.prepare_production.source_files", return_value=([path.name], "mit")), \
                    patch("data.prepare_production.stream_file", return_value=file_rows):
                try:
                    try:
                        sample_tokenizer({"sources": [spec]}, root, set(), "recipe", 0)
                    except TimeoutError:
                        # Keep the failure traceback alive: it must not own a live
                        # SQLite write transaction or an open source file.
                        with self.assertRaises(ValueError):
                            file_rows.handle.read()
                        connection = sqlite3.connect(root / "tokenizer-sample/sample.sqlite3", timeout=0)
                        try:
                            connection.execute("BEGIN EXCLUSIVE")
                            connection.rollback()
                        finally:
                            connection.close()
                    else:
                        self.fail("Sampling must respect its deadline")
                finally:
                    file_rows.handle.close()

    @unittest.skipUnless(Path("/kaggle/working").is_dir(), "Integration executes on Kaggle only")
    def test_tokenizer_sample_cap_releases_source_before_fitting_finishes(self):
        with tempfile.TemporaryDirectory(dir="/kaggle/working") as directory:
            root = Path(directory)
            _, _, config, _ = _preparation_fixture(root)
            config["vocab_size"] = 300
            spec = config["sources"][0]
            spec["tokenizer_chars"] = 4000
            path = root / "rows.jsonl"
            repo = next(f"repo-{i}" for i in range(100) if document_split(
                {"repo_name": f"repo-{i}"}, "", 42, .01) == "train")
            path.write_text("".join(json.dumps({
                "text": f"def candidate_{i}(number):\n    return number + {i}\n",
                "repo_name": repo}) + "\n" for i in range(200)), encoding="utf-8")
            file_rows = _JsonFileRows(path)
            output = root / "sampling"
            output.mkdir()
            with patch("data.prepare_production.source_files", return_value=([path.name], "mit")), \
                    patch("data.prepare_production.stream_file", return_value=file_rows):
                try:
                    sample_tokenizer(config, output, set(), "recipe", time.monotonic() + 600)
                    with self.assertRaises(ValueError):
                        file_rows.handle.read()
                    receipt = json.loads((output / "tokenizer-sample.json").read_text())
                    self.assertGreaterEqual(receipt["sources"]["fixture"]["characters"], 4000)
                    self.assertLess(receipt["sources"]["fixture"]["documents"], 200)
                finally:
                    file_rows.handle.close()

    @unittest.skipUnless(Path("/kaggle/working").is_dir(), "Integration executes on Kaggle only")
    def test_finished_signal_follows_partial_stream_resource_release(self):
        with tempfile.TemporaryDirectory(dir="/kaggle/working") as directory:
            root = Path(directory)
            first, config_path, _, rows = _preparation_fixture(root)
            path = root / "rows.jsonl"
            path.write_text("".join(json.dumps(record[0]) + "\n" for record in
                                    rows({}, {"file": 0, "row": 0})), encoding="utf-8")
            file_rows = _JsonFileRows(path)

            class Output(io.StringIO):
                open_files_at_finish = None

                def write(self, text):
                    if text.startswith("PREPARATION_SESSION_FINISHED "):
                        self.open_files_at_finish = int(not file_rows.handle.closed)
                    return super().write(text)

            output = Output()
            with patch("data.prepare_production.build_benchmark_index", return_value=(set(), [])), \
                    patch("data.prepare_production.source_files", return_value=([path.name], "mit")), \
                    patch("data.prepare_production.stream_file", return_value=file_rows), \
                    redirect_stdout(output):
                try:
                    result = prepare(config_path, first)
                    self.assertEqual(result["source_progress"]["fixture"]["row"], 2)
                    self.assertEqual(output.open_files_at_finish, 0,
                                     "The completion signal must follow stream cleanup")
                finally:
                    file_rows.handle.close()

    @unittest.skipUnless(Path("/kaggle/working").is_dir(), "Integration executes on Kaggle only")
    def test_resume_keeps_cursor_and_cross_session_duplicate_index(self):
        with tempfile.TemporaryDirectory(dir="/kaggle/working") as directory:
            root = Path(directory)
            first, config_path, config, rows = _preparation_fixture(root)
            second = root / "second"

            with patch("data.prepare_production.build_benchmark_index", return_value=(set(), [])), \
                    patch("data.prepare_production.source_rows", side_effect=rows):
                a = prepare(config_path, first)
                b = prepare(config_path, second, first)
                self.assertEqual(a["source_progress"]["fixture"]["documents"], 2)
                self.assertEqual(b["source_progress"]["fixture"]["documents"], 3)
                self.assertEqual(b["source_progress"]["fixture"]["counts"]["exact_duplicate"], 1)
                self.assertEqual(b["source_progress"]["fixture"]["row"], 4)
                self.assertEqual(a["tokenizer"], b["tokenizer"])
                changed = json.loads(config_path.read_text())
                changed["seed"] = 99
                config_path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "configuration/checkpoint"):
                    prepare(config_path, root / "wrong-config", second)

    @unittest.skipUnless(Path("/kaggle/working").is_dir(), "Integration executes on Kaggle only")
    def test_stream_failure_preserves_last_commit_and_resumes_without_skipping(self):
        with tempfile.TemporaryDirectory(dir="/kaggle/working") as directory:
            root = Path(directory)
            first, config_path, config, rows = _preparation_fixture(root)
            config["session_max_tokens"] = 100000
            config_path.write_text(json.dumps(config))

            def interrupted_rows(spec, cursor):
                for record in rows(spec, cursor):
                    if cursor["row"] > 2:
                        raise TimeoutError("Simulated source transport interruption")
                    yield record

            with patch("data.prepare_production.build_benchmark_index", return_value=(set(), [])):
                with patch("data.prepare_production.source_rows", side_effect=interrupted_rows):
                    with self.assertRaises(TimeoutError):
                        prepare(config_path, first)
                receipt_path = first / "preparation-manifest.json"
                self.assertTrue(receipt_path.is_file(), "Committed work needs a resumable receipt after failure")
                receipt = json.loads(receipt_path.read_text())
                self.assertEqual(receipt["status"], "interrupted-error")
                self.assertEqual(receipt["source_progress"]["fixture"]["row"], 2)
                self.assertEqual(receipt["state_sha256"], file_sha256(first / "state.sqlite3"))
                with patch("data.prepare_production.source_rows", side_effect=rows):
                    resumed = prepare(config_path, root / "resumed", first)
                self.assertEqual(resumed["source_progress"]["fixture"]["documents"], 3)
                self.assertEqual(resumed["source_progress"]["fixture"]["counts"]["exact_duplicate"], 1)

    @unittest.skipUnless(Path("/kaggle/working").is_dir(), "Integration executes on Kaggle only")
    def test_resume_rejects_a_different_preparation_pipeline_identity(self):
        with tempfile.TemporaryDirectory(dir="/kaggle/working") as directory:
            root = Path(directory)
            first, config_path, _, rows = _preparation_fixture(root)
            with patch("data.prepare_production.build_benchmark_index", return_value=(set(), [])), \
                    patch("data.prepare_production.source_rows", side_effect=rows):
                prepare(config_path, first)
                receipt_path = first / "preparation-manifest.json"
                receipt = json.loads(receipt_path.read_text())
                receipt["pipeline_identity"] = {"sha256": "not-the-current-pipeline"}
                receipt_path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, "pipeline identity"):
                    prepare(config_path, root / "changed-code", first)

    @unittest.skipUnless(Path("/kaggle/working").is_dir(), "Integration executes on Kaggle only")
    def test_atomically_promoted_tokenizer_recovers_missing_receipt_without_replacing_it(self):
        with tempfile.TemporaryDirectory(dir="/kaggle/working") as directory:
            root = Path(directory)
            first, config_path, config, rows = _preparation_fixture(root)
            config["vocab_size"] = 300
            config_path.write_text(json.dumps(config))
            sample = first / "tokenizer-sample/foundation_dedup.jsonl"
            sample.write_text("".join(json.dumps({"text": f"def candidate_{i}(number):\n    return number + {i}\n"})
                                      + "\n" for i in range(50)))
            train_custom_bpe_tokenizer(str(sample.parent), str(first / "tokenizer"), 300)
            before = tokenizer_identity(first / "tokenizer")
            (first / "tokenizer-identity.json").unlink()
            (first / "tokenizer-sample.json").write_text(json.dumps({
                "config_sha256": file_sha256(config_path), "sha256": file_sha256(sample),
                "train_only": True, "sources": {}}))
            with patch("data.prepare_production.build_benchmark_index", return_value=(set(), [])), \
                    patch("data.prepare_production.source_rows", side_effect=rows):
                result = prepare(config_path, first)
            self.assertEqual(result["tokenizer"], before)
            self.assertGreater(result["source_progress"]["fixture"]["documents"], 0)

    @unittest.skipUnless(Path("/kaggle/working").is_dir(), "Integration executes on Kaggle only")
    def test_missing_external_manifest_recovers_atomic_window_checkpoint(self):
        with tempfile.TemporaryDirectory(dir="/kaggle/working") as directory:
            root = Path(directory)
            first, config_path, _, rows = _preparation_fixture(root)
            with patch("data.prepare_production.build_benchmark_index", return_value=(set(), [])), \
                    patch("data.prepare_production.source_rows", side_effect=rows):
                original = prepare(config_path, first)
                (first / "preparation-manifest.json").unlink()
                resumed = prepare(config_path, root / "recovered", first)
            self.assertTrue(resumed["resumed_from_atomic_checkpoint"])
            self.assertEqual(resumed["tokenizer"], original["tokenizer"])
            self.assertEqual(resumed["source_progress"]["fixture"]["documents"], 3)

    @unittest.skipUnless(Path("/kaggle/working").is_dir(), "Integration executes on Kaggle only")
    def test_hot_journal_is_rolled_back_before_snapshot_validation(self):
        with tempfile.TemporaryDirectory(dir="/kaggle/working") as directory:
            root = Path(directory)
            first, config_path, _, rows = _preparation_fixture(root)
            with patch("data.prepare_production.build_benchmark_index", return_value=(set(), [])), \
                    patch("data.prepare_production.source_rows", side_effect=rows):
                prepare(config_path, first)
                crash = subprocess.run([sys.executable, "-c",
                    "import os,sqlite3,sys; db=sqlite3.connect(sys.argv[1]); "
                    "db.execute('PRAGMA cache_size=1'); db.execute('BEGIN IMMEDIATE'); "
                    "db.executemany('INSERT INTO documents VALUES (?)', "
                    "((f'uncommitted-{i}',) for i in range(500))); os._exit(73)",
                    str(first / "state.sqlite3")], capture_output=True, timeout=30)
                self.assertEqual(crash.returncode, 73)
                self.assertTrue((first / "state.sqlite3-journal").is_file())
                resumed = prepare(config_path, root / "recovered-journal", first)
            self.assertEqual(resumed["source_progress"]["fixture"]["documents"], 3)
            self.assertEqual(resumed["source_progress"]["fixture"]["counts"]["exact_duplicate"], 1)


if __name__ == "__main__":
    unittest.main()
