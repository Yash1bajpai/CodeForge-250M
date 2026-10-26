"""Kaggle CPU preparation with immutable windows and a disk-backed resume index."""
import argparse
import ast
from collections import Counter
from contextlib import contextmanager, ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sqlite3
import sys
import time

import yaml

from data.prepare_fresh import (build_benchmark_index, file_sha256, lexical_windows,
                                source_files, stream_file)
from data.instruction_format import CONTROL, normalize_instruction
from data.tokenize_dataset import apply_fim_transformation
from data.train_tokenizer import train_custom_bpe_tokenizer


@contextmanager
def _closing_resource(resource):
    """Close owned resources without replacing an in-flight processing failure."""
    try:
        yield resource
    finally:
        failure = sys.exc_info()[1]
        close = getattr(resource, "close", None)
        if close is not None:
            try:
                close()
            except BaseException as cleanup_failure:
                if failure is None or isinstance(failure, GeneratorExit):
                    raise
                failure.add_note("Resource cleanup also failed: " + type(cleanup_failure).__name__)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def document_key(text):
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def document_split(row, text, seed, fraction):
    # Repository grouping prevents related files from entering both splits.
    group = row.get("repo_name") or document_key(text)
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return "validation" if int.from_bytes(digest[:8], "big") < fraction * 2**64 else "train"


def repeated_blocks(text):
    # Bounded, conservative repeated-span filter, NOT exhaustive near-dedup.
    words = text.split()
    if len(words) < 256:
        return []
    starts = sorted({int(i * (len(words) - 256) / 7) for i in range(8)})
    return [hashlib.sha256(" ".join(words[i:i + 256]).encode()).digest()
            for i in starts if len(set(words[i:i + 256])) >= 32]


def tokenizer_identity(directory):
    files = {p.name: file_sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}
    return {"files": files, "sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()).hexdigest()}


def _json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def pipeline_identity():
    directory = Path(__file__).resolve().parent
    files = {name: file_sha256(directory / name) for name in (
        "prepare_production.py", "prepare_fresh.py", "instruction_format.py",
        "train_tokenizer.py", "tokenize_dataset.py")}
    software = {name: importlib.metadata.version(name) for name in (
        "datasets", "tokenizers", "transformers", "torch", "pyarrow", "pyyaml")}
    software["python"] = platform.python_version()
    identity = {"files": files, "software": software}
    return {**identity, "sha256": _json_sha256(identity)}


def validate_tokenizer(directory, vocab_size):
    from tokenizers import Tokenizer, pre_tokenizers
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    required = ["<|endoftext|>", "<|unk|>", "<|pad|>", "<|fim_prefix|>", "<|fim_middle|>",
                "<|fim_suffix|>", "<|tool_call|>", "<|tool_result|>", "<|thinking|>",
                "<|json_start|>", "<|json_end|>"]
    ids = [tokenizer.token_to_id(token) for token in required]
    if (tokenizer.get_vocab_size() != vocab_size or None in ids or len(set(ids)) != len(required)
            or any(len(tokenizer.encode(token).ids) != 1 for token in required)
            or not set(pre_tokenizers.ByteLevel.alphabet()).issubset(tokenizer.get_vocab())):
        raise ValueError("Production tokenizer vocabulary/special token gate failed")


def read_checkpoint(directory, connection, config_hash, pipeline):
    """Recover only committed SQLite metadata; never accept an unverified prefix."""
    if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise ValueError("Checkpoint database integrity check failed")
    rows = dict(connection.execute("SELECT key, value FROM meta WHERE key IN ('state', 'checkpoint')"))
    if "state" not in rows or "checkpoint" not in rows:
        raise ValueError("Checkpoint lacks atomic preparation metadata; old snapshots need an explicit audit")
    state, checkpoint = json.loads(rows["state"]), json.loads(rows["checkpoint"])
    if checkpoint.get("format_version") != 2 or checkpoint["config_sha256"] != config_hash:
        raise ValueError("Resume configuration/checkpoint checksum mismatch")
    if checkpoint.get("pipeline_identity") != pipeline or state.get("pipeline_identity") != pipeline:
        raise ValueError("Resume pipeline identity mismatch")
    if (checkpoint["state_json_sha256"] != _json_sha256(state)
            or checkpoint["last_global_window"] != state["window"]
            or checkpoint["source_progress"] != state["sources"]
            or checkpoint["config_sha256"] != state["config_sha256"]):
        raise ValueError("Committed checkpoint metadata is inconsistent")
    identity = checkpoint["tokenizer"]
    if (tokenizer_identity(directory / "tokenizer") != identity
            or state.get("tokenizer_sha256") != identity["sha256"]):
        raise ValueError("Resume tokenizer changed")
    if connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] != sum(
            source["documents"] for source in state["sources"].values()):
        raise ValueError("Checkpoint duplicate index and document counts differ")
    external_path = directory / "preparation-manifest.json"
    recovered = True
    if external_path.is_file():
        external = json.loads(external_path.read_text())
        if external["config_sha256"] != config_hash:
            raise ValueError("Resume configuration/checkpoint checksum mismatch")
        if external.get("pipeline_identity") != pipeline:
            raise ValueError("Resume pipeline identity mismatch")
        if external["tokenizer"] != identity or external["last_global_window"] > state["window"]:
            raise ValueError("Published checkpoint is inconsistent")
        if external["last_global_window"] == state["window"]:
            if external["state_json_sha256"] != checkpoint["state_json_sha256"]:
                raise ValueError("Published and atomic checkpoint states differ")
            if external.get("state_sha256"):
                database = Path(connection.execute("PRAGMA database_list").fetchone()[2])
                if file_sha256(database) != external["state_sha256"]:
                    raise ValueError("Resume configuration/checkpoint checksum mismatch")
                recovered = False
    for item in checkpoint["owned_windows"]:
        if not 0 < item["number"] <= state["window"]:
            raise ValueError("Invalid committed window number")
        window = directory / "windows" / f"window-{item['number']:06d}"
        if file_sha256(window / "manifest.json") != item["manifest_sha256"]:
            raise ValueError("Committed window manifest changed")
        verify_window(window)
    return state, checkpoint, recovered


def admit(row, spec, config, exclusion, corpus_license):
    if not isinstance(row, dict):
        return None, "invalid_record"
    license_value = row.get(spec["license_field"]) if spec.get("license_field") else corpus_license
    if license_value not in spec["allowed_licenses"]:
        return None, "license_rejected"
    language = row.get(spec.get("language_field", "language"))
    if spec.get("allowed_languages") and language not in spec["allowed_languages"]:
        return None, "language_rejected"
    try:
        record = normalize_instruction(row, spec["format"]) if spec.get("format") else {
            "text": row.get(spec["text_field"]), "assistant_spans": None}
    except ValueError:
        return None, "format_rejected"
    text = record["text"]
    if not isinstance(text, str) or not 30 <= len(text) <= config["max_document_chars"]:
        return None, "size_rejected"
    if record["assistant_spans"] is None and any(token in text for token in CONTROL):
        return None, "control_token_rejected"
    if sum(c.isalnum() for c in text) < len(text) * .25:
        return None, "quality_rejected"
    if spec.get("python_ast") and (language in (None, "Python")):
        try:
            ast.parse(text)
        except (SyntaxError, ValueError, RecursionError):
            return None, "python_ast_rejected"
    policy = config["benchmark_exclusion"]
    if any(window in exclusion for window in lexical_windows(
            text, policy["window_tokens"], policy["minimum_distinct_tokens"])):
        return None, "benchmark_rejected"
    record.update(license=license_value, language=language)
    return record, None

def path_language(name):
    # The pinned GitHub parquet schema stores content/path, not code/language.
    extensions = {".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
                  ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell", ".command": "Shell",
                  ".ps1": "PowerShell", ".psd1": "PowerShell", ".psm1": "PowerShell",
                  ".c": "C", ".h": "C", ".C": "C++", ".H": "C++", ".cpp": "C++",
                  ".hpp": "C++", ".cc": "C++", ".hh": "C++", ".c++": "C++", ".h++": "C++",
                  ".go": "GO", ".rs": "Rust", ".java": "Java", ".cs": "C#", ".sql": "SQL",
                  ".html": "HTML", ".css": "CSS", ".dockerfile": "Dockerfile",
                  ".md": "Markdown", ".markdown": "Markdown"}
    if not isinstance(name, str):
        return None
    if name.endswith("Dockerfile"):
        return "Dockerfile"
    return extensions.get(Path(name).suffix)



def source_rows(spec, cursor):
    """Keep this iterator alive between commits; replay only once per session."""
    files, license_value = source_files(spec)
    if not spec.get("license_field"):
        licenses = license_value if isinstance(license_value, list) else [license_value]
        matching = [value for value in licenses if value in spec["allowed_licenses"]]
        if not matching:
            raise ValueError("Unapproved corpus license: " + spec["name"])
        license_value = matching[0]
    while cursor["file"] < len(files):
        filename = files[cursor["file"]]
        skipped = cursor["row"]
        with ExitStack() as resources:
            stream = resources.enter_context(_closing_resource(stream_file(spec, filename)))
            rows = iter(stream)
            if rows is not stream:
                resources.enter_context(_closing_resource(rows))
            for index, row in enumerate(rows):
                if index < skipped:
                    continue
                if index == 0:
                    print("SOURCE_SCHEMA " + json.dumps({"source": spec["name"], "fields": sorted(row)}), flush=True)
                if not spec.get("format") and spec["text_field"] not in row:
                    raise ValueError("Source is missing configured text field: " + spec["name"])
                if spec.get("license_field") and spec["license_field"] not in row:
                    raise ValueError("Source is missing record-license field: " + spec["name"])
                if spec.get("language_from_path"):
                    row = {**row, "language": path_language(row.get("path"))}
                cursor["row"] = index + 1
                yield row, filename, index, license_value
        cursor["file"] += 1
        cursor["row"] = 0


def sample_tokenizer(config, output, exclusion, config_hash, deadline):
    sample_dir = output / "tokenizer-sample"
    sample_dir.mkdir(exist_ok=True)
    sample_path = sample_dir / "foundation_dedup.jsonl"
    receipt_path = output / "tokenizer-sample.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["config_sha256"] != config_hash or receipt["sha256"] != file_sha256(sample_path):
            raise ValueError("Tokenizer sample changed")
    else:
        temporary = sample_path.with_suffix(".part")
        with _closing_resource(sqlite3.connect(sample_dir / "sample.sqlite3")) as sample_db:
            sample_db.execute("CREATE TABLE IF NOT EXISTS hashes (digest TEXT PRIMARY KEY)")
            sample_db.execute("DELETE FROM hashes")
            counts = {}
            with temporary.open("w", encoding="utf-8") as handle:
                for spec in config["sources"]:
                    cap = spec.get("tokenizer_chars", 0)
                    if not cap:
                        continue
                    print("TOKENIZER_SAMPLE source=" + spec["name"], flush=True)
                    chars = documents = 0
                    with _closing_resource(source_rows(spec, {"file": 0, "row": 0})) as rows:
                        for row, _, _, license_value in rows:
                            if time.monotonic() >= deadline - 300:
                                raise TimeoutError("Tokenizer sampling deadline; rerun from saved output")
                            record, _ = admit(row, spec, config, exclusion, license_value)
                            if record is None or document_split(row, record["text"], config["seed"],
                                                               config["validation_fraction"]) != "train":
                                continue
                            digest = document_key(record["text"])
                            if not sample_db.execute("INSERT OR IGNORE INTO hashes VALUES (?)", (digest,)).rowcount:
                                continue
                            handle.write(json.dumps({"text": record["text"]}, ensure_ascii=False) + "\n")
                            chars += len(record["text"])
                            documents += 1
                            if chars >= cap:
                                break
                    if chars < cap:
                        raise ValueError("Insufficient representative tokenizer sample: " + spec["name"])
                    counts[spec["name"]] = {"characters": chars, "documents": documents}
        os.replace(temporary, sample_path)
        atomic_json(receipt_path, {"config_sha256": config_hash, "sha256": file_sha256(sample_path),
                                   "train_only": True, "sources": counts})
    print("TOKENIZER_FIT vocabulary=32000 training_documents_only=true", flush=True)
    temporary_tokenizer = output / "tokenizer-building"
    if temporary_tokenizer.exists():
        shutil.rmtree(temporary_tokenizer)
    train_custom_bpe_tokenizer(str(sample_dir), str(temporary_tokenizer), config["vocab_size"])
    validate_tokenizer(temporary_tokenizer, config["vocab_size"])
    os.replace(temporary_tokenizer, output / "tokenizer")
    identity = tokenizer_identity(output / "tokenizer")
    atomic_json(output / "tokenizer-identity.json", identity)
    print("TOKENIZER_FROZEN sha256=" + identity["sha256"], flush=True)
    return identity


def encode_record(tokenizer, record, spec, key, seq_len):
    text = record["text"]
    spans = record["assistant_spans"]
    if spans is None:
        rate = spec.get("fim_rate", 0)
        if spec.get("fim_languages") and record.get("language") not in spec["fim_languages"]:
            rate = 0
        if rate:
            text = apply_fim_transformation(text, rate, random.Random(int(key[:16], 16)))
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        ids.append(tokenizer.token_to_id("<|endoftext|>"))
        return ids, [1] * len(ids)
    encoding = tokenizer.encode(text, add_special_tokens=False)
    if len(encoding.ids) > seq_len:
        return None, None
    mask = [int(any(start >= low and end <= high and end > start for low, high in spans))
            for start, end in encoding.offsets]
    if not any(mask[1:]):
        return None, None
    return encoding.ids, mask


class WindowWriter:
    def __init__(self, root, number, config, tokenizer_hash):
        self.root, self.number, self.config = root, number, config
        self.path = root / "windows" / f"window-{number:06d}.part"
        self.destination = self.path.with_suffix("")
        # Only the next uncommitted generation can be an orphan of this writer.
        for path in (self.path, self.destination):
            if path.exists():
                shutil.rmtree(path)
        self.path.mkdir(parents=True)
        self.text = (self.path / "documents.jsonl").open("w", encoding="utf-8")
        self.buffers, self.shards, self.pending = {}, [], {}
        self.manifest = {"window": number, "tokenizer_sha256": tokenizer_hash,
                         "seq_len": config["seq_length"], "shards": self.shards,
                         "valid_tokens": 0, "padding_tokens": 0, "loss_tokens": 0,
                         "discarded_tokens": 0, "documents": 0}

    def row(self, key, ids, mask):
        length = len(ids)
        if length < 2:
            self.manifest["discarded_tokens"] += length
            return
        mask[0] = 0  # A row's first token has no preceding token to predict it.
        pad = self.config["seq_length"] - length
        ids = ids + [0] * pad
        mask = mask + [0] * pad
        buffer = self.buffers.setdefault(key, [])
        buffer.append((ids, mask, length))
        if len(buffer) >= self.config["shard_sequences"]:
            self.flush(key)

    def add(self, key, ids, mask, record):
        self.text.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.manifest["documents"] += 1
        if record["stage"] != "foundation":
            self.row(key, ids, mask)
            return
        pending_ids, pending_mask = self.pending.setdefault(key, ([], []))
        offset = 0
        while offset < len(ids):
            count = min(self.config["seq_length"] - len(pending_ids), len(ids) - offset)
            pending_ids.extend(ids[offset:offset + count])
            pending_mask.extend(mask[offset:offset + count])
            offset += count
            if len(pending_ids) == self.config["seq_length"]:
                self.row(key, pending_ids[:], pending_mask[:])
                pending_ids.clear()
                pending_mask.clear()

    def flush(self, key):
        import torch
        buffer = self.buffers[key]
        if not buffer:
            return
        relative = Path(*key)
        directory = self.path / relative
        directory.mkdir(parents=True, exist_ok=True)
        index = len(self.shards) + 1
        files = {"file": f"shard_{index:06d}.pt", "valid_file": f"valid_{index:06d}.pt",
                 "loss_file": f"loss_{index:06d}.pt"}
        payloads = {"file": torch.tensor([row[0] for row in buffer], dtype=torch.uint16),
                    "valid_file": {"valid_len": torch.tensor([row[2] for row in buffer], dtype=torch.int32)},
                    "loss_file": torch.tensor([row[1] for row in buffer], dtype=torch.uint8)}
        item = {"source": key[1], "stage": key[0], "split": key[2], "rows": len(buffer),
                "valid_tokens": sum(row[2] for row in buffer),
                "padding_tokens": sum(self.config["seq_length"] - row[2] for row in buffer),
                "loss_tokens": sum(sum(row[1]) for row in buffer)}
        for field, filename in files.items():
            path = directory / filename
            with path.open("xb") as handle:
                torch.save(payloads[field], handle)
                handle.flush()
                os.fsync(handle.fileno())
            item[field] = str(relative / filename).replace("\\", "/")
            item[field + "_sha256"] = file_sha256(path)
        self.shards.append(item)
        for field in ("valid_tokens", "padding_tokens", "loss_tokens"):
            self.manifest[field] += item[field]
        buffer.clear()

    def finish(self):
        for key, (ids, mask) in self.pending.items():
            if ids:
                self.row(key, ids, mask)
        for key in self.buffers:
            self.flush(key)
        self.text.flush()
        os.fsync(self.text.fileno())
        self.text.close()
        self.manifest["documents_sha256"] = file_sha256(self.path / "documents.jsonl")
        atomic_json(self.path / "manifest.json", self.manifest)
        os.replace(self.path, self.destination)
        return self.manifest


def verify_window(path):
    import torch
    manifest = json.loads((path / "manifest.json").read_text())
    actual = Counter()
    if file_sha256(path / "documents.jsonl") != manifest["documents_sha256"]:
        raise ValueError("Provenance checksum mismatch")
    for item in manifest["shards"]:
        for field in ("file", "valid_file", "loss_file"):
            if file_sha256(path / item[field]) != item[field + "_sha256"]:
                raise ValueError("Shard checksum mismatch")
        ids = torch.load(path / item["file"], weights_only=True, map_location="cpu")
        valid = torch.load(path / item["valid_file"], weights_only=True, map_location="cpu")["valid_len"]
        loss = torch.load(path / item["loss_file"], weights_only=True, map_location="cpu")
        if (ids.dtype != torch.uint16 or loss.dtype != torch.uint8 or ids.shape != loss.shape
                or ids.shape != (len(valid), manifest["seq_len"]) or int(valid.min()) < 2
                or int(valid.max()) > manifest["seq_len"] or int(loss.max()) > 1
                or bool(loss[:, 0].any())):
            raise ValueError("Invalid token/length/loss layout")
        for index, length in enumerate(valid.tolist()):
            if bool(loss[index, length:].any()):
                raise ValueError("Padding has supervised targets")
        measured = {"valid_tokens": int(valid.sum()), "loss_tokens": int(loss.sum()),
                    "padding_tokens": ids.numel() - int(valid.sum())}
        for field, count in measured.items():
            if count != item[field]:
                raise ValueError("Shard accounting mismatch")
            actual[field] += count
    if any(actual[field] != manifest[field] for field in ("valid_tokens", "padding_tokens", "loss_tokens")):
        raise ValueError("Window accounting mismatch")
    return manifest


def prepare(config_path, output, resume_from=None):
    output = Path(output).resolve()
    kaggle = Path("/kaggle/working")
    if not kaggle.is_dir() or not output.is_relative_to(kaggle.resolve()):
        raise RuntimeError("Production data processing is Kaggle-only; the PC is a controller")
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    if config.get("purpose") != "data-preparation-only":
        raise ValueError("Preparation purpose must be explicit")
    sources = config["sources"]
    names = [spec["name"] for spec in sources]
    if len(names) != len(set(names)) or sum(s.get("tokenizer_chars", 0) for s in sources) != config["tokenizer_sample_chars"]:
        raise ValueError("Invalid source names or tokenizer sample allocation")
    config_hash = file_sha256(config_path)
    pipeline = pipeline_identity()
    token_limit = os.environ.get("CF_PREP_SESSION_TOKEN_LIMIT")
    if token_limit is not None:
        token_limit = int(token_limit)
        if token_limit < 1:
            raise ValueError("Session token limit must be positive")
        config["session_max_tokens"] = token_limit
    deadline = time.monotonic() + config["session_seconds"]
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "writer.lock"
    # Atomic exclusive create prevents concurrent notebook writers.
    if lock.exists():
        prior_pid = int(lock.read_text())
        try:
            os.kill(prior_pid, 0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise RuntimeError("Preparation writer is already active")
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    connection = None
    state = identity = writer = None
    exclusion, benchmarks = set(), []
    owned_windows, parent_reference = [], None
    resumed_from_atomic_checkpoint = False
    checkpoint_ready = False
    stream_resources = ExitStack()

    def checkpoint_payload(status, database_hash=False, failure=None):
        payload = {
            "format_version": 2, "run_id": config["run_id"], "status": status,
            "resumed_from_atomic_checkpoint": resumed_from_atomic_checkpoint,
            "config_sha256": config_hash, "pipeline_identity": pipeline,
            "state_json_sha256": _json_sha256(state),
            "state_sha256": file_sha256(output / "state.sqlite3") if database_hash else None,
            "tokenizer": identity, "source_progress": state["sources"],
            "last_global_window": state["window"], "owned_windows": owned_windows,
            "parent": parent_reference, "unique_data_not_training_epochs": True,
            "model_training_started": False, "training_consumer_ready": False,
            "training_consumer_gate": "Validate assistant-loss-mask consumption before SFT training",
            "benchmark_exclusion": {"sources": benchmarks, "windows": len(exclusion), "complete": False},
            "deduplication": "Global normalized exact SHA256 + eight sampled 256-word repeated blocks; not exhaustive near-dedup",
            "budget_semantics": "Per-source training-input targets; whole-document overshoot bounded by max_document_chars; validation extra; no repetition to fill shortfalls",
            "config": config}
        if failure is not None:
            payload["failure"] = {"type": type(failure).__name__}
        return payload

    def commit_checkpoint(status):
        # The cursor, duplicate index and recovery receipt share ONE transaction.
        payload = checkpoint_payload(status)
        connection.execute("INSERT OR REPLACE INTO meta VALUES ('state', ?)", (json.dumps(state),))
        connection.execute("INSERT OR REPLACE INTO meta VALUES ('checkpoint', ?)", (json.dumps(payload),))
        connection.commit()
        atomic_json(output / "preparation-manifest.json", payload)
    try:
        prior = Path(resume_from) if resume_from is not None else None
        if prior is not None:
            if (output / "state.sqlite3").exists():
                raise FileExistsError("Resume destination already has state")
            shutil.copyfile(prior / "state.sqlite3", output / "state.sqlite3")
            journal = prior / "state.sqlite3-journal"
            if journal.is_file():
                # SQLite rolls an interrupted write back before metadata is read.
                shutil.copyfile(journal, output / "state.sqlite3-journal")
        connection = sqlite3.connect(output / "state.sqlite3")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA cache_size=-32768")
        connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS documents (digest TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE IF NOT EXISTS blocks (digest BLOB PRIMARY KEY)")
        saved = connection.execute("SELECT value FROM meta WHERE key='state'").fetchone()
        has_checkpoint = connection.execute("SELECT 1 FROM meta WHERE key='checkpoint'").fetchone()
        if prior is not None or (saved and has_checkpoint):
            state, previous_checkpoint, recovered = read_checkpoint(
                prior or output, connection, config_hash, pipeline)
            resumed_from_atomic_checkpoint = recovered
            identity = previous_checkpoint["tokenizer"]
            if prior is not None:
                shutil.copytree(prior / "tokenizer", output / "tokenizer")
                atomic_json(output / "tokenizer-identity.json", identity)
                parent_reference = {
                    "kernel": os.environ.get("CF_PREP_PARENT_KERNEL"),
                    "checkpoint_sha256": _json_sha256(previous_checkpoint),
                    "manifest_sha256": file_sha256(prior / "preparation-manifest.json")
                    if (prior / "preparation-manifest.json").is_file() else None,
                    "previous_run_id": previous_checkpoint["run_id"],
                    "last_global_window": state["window"]}
                atomic_json(output / "parent.json", parent_reference)
            else:
                owned_windows = list(previous_checkpoint["owned_windows"])
                parent_reference = previous_checkpoint.get("parent")
            print("CHECKPOINT_VERIFIED " + json.dumps({
                "window": state["window"], "recovered_atomic_state": recovered}), flush=True)
        else:
            state = json.loads(saved[0]) if saved else {
                "config_sha256": config_hash, "pipeline_identity": pipeline, "window": 0,
                "sources": {name: {"file": 0, "row": 0, "tokens": 0, "train_tokens": 0,
                                    "validation_tokens": 0, "documents": 0, "exhausted": False,
                                    "counts": {}} for name in names}}
            if state["config_sha256"] != config_hash:
                raise ValueError("Existing output belongs to a different configuration")
            if state.get("pipeline_identity") != pipeline:
                raise ValueError("Existing output pipeline identity mismatch")
            if state["window"] or state.get("sample_indexed"):
                raise ValueError("Committed state lacks its atomic recovery receipt")
            connection.execute("INSERT OR REPLACE INTO meta VALUES ('state', ?)", (json.dumps(state),))
            connection.commit()
        print("BENCHMARK_EXCLUSION building pinned public reference index", flush=True)
        exclusion, benchmarks = build_benchmark_index(config["benchmark_exclusion"])
        if not (output / "tokenizer-identity.json").exists() and (output / "tokenizer").is_dir():
            if state["window"] or state.get("tokenizer_sha256"):
                raise ValueError("An already-consumed tokenizer lost its identity receipt")
            sample = output / "tokenizer-sample/foundation_dedup.jsonl"
            sample_receipt = json.loads((output / "tokenizer-sample.json").read_text())
            if (sample_receipt["config_sha256"] != config_hash
                    or sample_receipt["sha256"] != file_sha256(sample) or not sample_receipt["train_only"]):
                raise ValueError("Cannot recover an unverified tokenizer sample")
            validate_tokenizer(output / "tokenizer", config["vocab_size"])
            atomic_json(output / "tokenizer-identity.json", tokenizer_identity(output / "tokenizer"))
            print("TOKENIZER_RECEIPT_RECOVERED no_refit=true", flush=True)
        if (output / "tokenizer-identity.json").exists():
            identity = json.loads((output / "tokenizer-identity.json").read_text())
            if tokenizer_identity(output / "tokenizer") != identity:
                raise ValueError("Frozen tokenizer checksum mismatch")
        else:
            identity = sample_tokenizer(config, output, exclusion, config_hash, deadline)
        if state.get("tokenizer_sha256", identity["sha256"]) != identity["sha256"]:
            raise ValueError("Resume state and frozen tokenizer differ")
        state["tokenizer_sha256"] = identity["sha256"]
        connection.execute("CREATE TABLE IF NOT EXISTS sample_hashes (digest TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE IF NOT EXISTS sample_blocks (digest BLOB PRIMARY KEY)")
        if not state.get("sample_indexed"):
            with (output / "tokenizer-sample/foundation_dedup.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    text = json.loads(line)["text"]
                    connection.execute("INSERT OR IGNORE INTO sample_hashes VALUES (?)", (document_key(text),))
                    connection.executemany("INSERT OR IGNORE INTO sample_blocks VALUES (?)",
                                           ((block,) for block in repeated_blocks(text)))
            state["sample_indexed"] = True
        commit_checkpoint("tokenizer-ready")
        checkpoint_ready = True
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(output / "tokenizer/tokenizer.json"))
        streams = {s["name"]: stream_resources.enter_context(
            _closing_resource(source_rows(s, state["sources"][s["name"]]))) for s in sources}
        initial_tokens = sum(s["tokens"] for s in state["sources"].values())
        session_status = "resume-required"
        print("TOKENIZATION_STARTED frozen_tokenizer=" + identity["sha256"], flush=True)
        while time.monotonic() < deadline - 180:
            if sum(s["tokens"] for s in state["sources"].values()) - initial_tokens >= config["session_max_tokens"]:
                break
            if shutil.disk_usage(output).free < config["min_free_bytes"]:
                session_status = "disk-floor-resume-required"
                break
            available = [s for s in sources if not state["sources"][s["name"]]["exhausted"]
                         and state["sources"][s["name"]]["train_tokens"] < s["target_tokens"]]
            if not available:
                session_status = "sources-exhausted-or-budgets-reached"
                break
            spec = min(available, key=lambda s: state["sources"][s["name"]]["train_tokens"] / s["target_tokens"])
            cursor = state["sources"][spec["name"]]
            writer = WindowWriter(output, state["window"] + 1, config, identity["sha256"])
            counts = Counter(cursor["counts"])
            connection.execute("BEGIN IMMEDIATE")
            for _ in range(config["checkpoint_documents"]):
                if time.monotonic() >= deadline - 180:
                    break
                try:
                    row, filename, row_index, license_value = next(streams[spec["name"]])
                except StopIteration:
                    cursor["exhausted"] = True
                    break
                counts["rows_examined"] += 1
                record, rejected = admit(row, spec, config, exclusion, license_value)
                if record is None:
                    counts[rejected] += 1
                    continue
                key = document_key(record["text"])
                blocks = repeated_blocks(record["text"])
                if connection.execute("SELECT 1 FROM documents WHERE digest=?", (key,)).fetchone():
                    counts["exact_duplicate"] += 1
                    continue
                if any(connection.execute("SELECT 1 FROM blocks WHERE digest=?", (block,)).fetchone() for block in blocks):
                    counts["repeated_block"] += 1
                    continue
                ids, mask = encode_record(tokenizer, record, spec, key, config["seq_length"])
                if ids is None:
                    counts["over_context_or_no_target"] += 1
                    continue
                split = document_split(row, record["text"], config["seed"], config["validation_fraction"])
                if split == "validation" and (
                        connection.execute("SELECT 1 FROM sample_hashes WHERE digest=?", (key,)).fetchone()
                        or any(connection.execute("SELECT 1 FROM sample_blocks WHERE digest=?", (b,)).fetchone()
                               for b in blocks)):
                    counts["tokenizer_validation_overlap_rejected"] += 1
                    continue
                provenance = {**record, "source": spec["name"], "stage": spec["stage"], "split": split,
                              "repo": spec["repo"], "revision": spec["revision"], "file": filename,
                              "row": row_index, "repo_name": row.get("repo_name"), "path": row.get("path"),
                              "url": row.get("url"), "normalized_sha256": key, "input_tokens": len(ids)}
                writer.add((spec["stage"], spec["name"], split), ids, mask, provenance)
                connection.execute("INSERT INTO documents VALUES (?)", (key,))
                connection.executemany("INSERT OR IGNORE INTO blocks VALUES (?)", ((b,) for b in blocks))
                cursor["tokens"] += len(ids)
                cursor[split + "_tokens"] += len(ids)
                cursor["documents"] += 1
                counts["accepted"] += 1
                if cursor["train_tokens"] >= spec["target_tokens"]:
                    break
            cursor["counts"] = dict(counts)
            window = writer.finish()
            verify_window(writer.destination)
            state["window"] += 1
            owned_windows.append({"number": state["window"],
                                  "manifest_sha256": file_sha256(writer.destination / "manifest.json")})
            commit_checkpoint("window-checkpoint")
            writer = None
            progress = {"status": "preparing", "window": state["window"], "source": spec["name"],
                        "session_input_tokens": sum(s["tokens"] for s in state["sources"].values()) - initial_tokens,
                        "sources": state["sources"], "last_window_valid_tokens": window["valid_tokens"],
                        "model_training_started": False}
            atomic_json(output / "progress.json", progress)
            print("PREPARATION_PROGRESS " + json.dumps({k: v for k, v in progress.items() if k != "sources"}), flush=True)
        # Finish the resource boundary before publishing successful completion.
        stream_resources.close()
        connection.close()
        connection = None
        manifest = checkpoint_payload(session_status, database_hash=True)
        atomic_json(output / "preparation-manifest.json", manifest)
        shutil.copyfile(config_path, output / "preparation-config.yaml")
        print("PREPARATION_SESSION_FINISHED " + json.dumps({"status": session_status,
              "input_tokens": sum(s["tokens"] for s in state["sources"].values()),
              "model_training_started": False}), flush=True)
        return manifest
    except BaseException as failure:
        # Preserve the error; publish only the last COMMITTED prefix for recovery.
        try:
            committed = None
            if connection is not None:
                connection.rollback()
                row = connection.execute("SELECT value FROM meta WHERE key='checkpoint'").fetchone()
                if row and checkpoint_ready:
                    committed = json.loads(row[0])
                    state = json.loads(connection.execute(
                        "SELECT value FROM meta WHERE key='state'").fetchone()[0])
                    owned_windows = committed["owned_windows"]
                    identity = committed["tokenizer"]
                connection.close()
                connection = None
            if writer is not None and committed is not None:
                if not writer.text.closed:
                    writer.text.close()
                if writer.number > state["window"]:
                    for orphan in (writer.path, writer.destination):
                        if orphan.exists():
                            shutil.rmtree(orphan)
            if committed is not None:
                atomic_json(output / "preparation-manifest.json",
                            checkpoint_payload("interrupted-error", database_hash=True, failure=failure))
        except Exception as checkpoint_failure:
            print("CHECKPOINT_PRESERVATION_FAILED type=" + type(checkpoint_failure).__name__, flush=True)
        raise
    finally:
        try:
            stream_resources.__exit__(*sys.exc_info())
        finally:
            if connection is not None:
                connection.close()
            lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    prepare(args.config, args.output, args.resume_from)


if __name__ == "__main__":
    main()
