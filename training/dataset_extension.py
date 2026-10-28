"""Explicit foundation-only handoff between disjoint production preparations."""
from contextlib import closing
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import torch

from training.production_data import LazyShardDataset, select_production_shards
from training.resume_safety import file_sha256, manifest_fingerprint, validate_resume_metadata


ORDER_VERSION = "extension-permutation-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def dataset_fingerprint(records, tokenizer_path, context, resume_state=None):
    """Old fingerprints are unchanged; extension identity also pins its handoff."""
    r = resume_state or {}
    if "extension" in r:
        entry, chain = r["extension"], r.get("lineage_chain", [])
        if (not chain or chain[-1] != {"transition": entry, "new_fingerprint": r.get("fingerprint")}
                or entry.get("parent_lineage_sha256") != _digest(chain[:-1])
                or any(right["transition"]["parent_fingerprint"] != left["new_fingerprint"]
                       for left, right in zip(chain, chain[1:]))
                or r.get("parent_fingerprint") != entry.get("parent_fingerprint")
                or r.get("new_fingerprint") != r.get("fingerprint")
                or r.get("consumed_old_set_identity") != entry.get("consumed_old_set_identity")
                or r.get("order_version") != ORDER_VERSION
                or r.get("order_seed") != entry.get("order_seed")):
            raise ValueError("Invalid dataset extension lineage/order")
        if context != r.get("identity_context"):
            raise ValueError("Extension config/stage/cumulative budget changed")
        if _digest(records["validation"]) != entry.get("validation_identity"):
            raise ValueError("Extension validation membership must remain frozen")
        if r.get("completed_updates") != r.get("optimizer_step_count"):
            raise ValueError("Extension applied-update/optimizer counters disagree")
        context = {**context, "extension": entry}
    return manifest_fingerprint(records, tokenizer_path, context)


def extension_position(dataset, resume_state):
    """Return segment-local cursor and cumulative input count at its start."""
    r = resume_state
    if "extension" not in r:
        return r["sample_cursor"], 0
    entry = r["extension"]
    start, tokens = entry["start_sample_cursor"], entry["start_input_tokens"]
    if (type(start) is not int or start < 0 or type(tokens) is not int or tokens < 0
            or not start <= r["sample_cursor"] <= start + len(dataset)
            or r["input_tokens_seen"] < tokens):
        raise ValueError("Invalid extension cumulative/segment counters")
    return r["sample_cursor"] - start, tokens


def extension_order(dataset, resume_state):
    seed = resume_state.get("order_seed")
    if (resume_state.get("order_version") != ORDER_VERSION or type(seed) is not int
            or not 0 <= seed < 2**63):
        raise ValueError("Invalid extension permutation identity/seed")
    return torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()


def _preparation(root):
    root = Path(root).resolve()
    path = root / "preparation-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    database = root / "state.sqlite3"
    if any(database.with_name(database.name + suffix).exists() for suffix in ("-journal", "-wal", "-shm")):
        raise ValueError("Extension requires a closed preparation database, without journal/WAL files")
    if (manifest.get("format_version") != 2 or not manifest.get("state_sha256")
            or file_sha256(database) != manifest["state_sha256"]):
        raise ValueError("Extension requires a hash-verified final preparation database/manifest")
    return manifest, file_sha256(path), database


def _verify_chain(new_root, parent, new, parent_hash, parent_db, new_db):
    reference = new.get("parent") or {}
    if (reference.get("manifest_sha256") != parent_hash
            or reference.get("last_global_window") != parent.get("last_global_window")
            or reference.get("previous_run_id") != parent.get("run_id")):
        raise ValueError("New preparation does not directly descend from the supplied parent")
    for key in ("run_id", "config_sha256", "pipeline_identity", "tokenizer"):
        if not parent.get(key) or new.get(key) != parent[key]:
            raise ValueError(f"Preparation {key} changed across extension")
    numbers = [item["number"] for item in new["owned_windows"]]
    last = parent["last_global_window"]
    if (type(last) is not int or not numbers
            or numbers != list(range(last + 1, last + 1 + len(numbers)))
            or new.get("last_global_window") != numbers[-1]):
        raise ValueError("Extension must own only contiguous NEW windows after the parent")
    # The verified global dedup index, not a claimed shard prefix, proves disjointness.
    # Preserve it transitively so a later extension cannot reintroduce older documents.
    uri = new_db.as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.execute("ATTACH DATABASE ? AS ancestor", (parent_db.as_uri() + "?mode=ro&immutable=1",))
        if db.execute("SELECT digest FROM ancestor.documents EXCEPT SELECT digest FROM main.documents LIMIT 1").fetchone():
            raise ValueError("New preparation lost the parent's global document index")
        seen = set()
        for number in numbers:
            documents = Path(new_root) / "windows" / f"window-{number:06d}" / "documents.jsonl"
            with documents.open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    key = hashlib.sha256(" ".join(record["text"].split()).encode("utf-8")).hexdigest()
                    if key != record.get("normalized_sha256"):
                        raise ValueError("New document normalized identity mismatch")
                    if key in seen or db.execute("SELECT 1 FROM ancestor.documents WHERE digest=?", (key,)).fetchone():
                        raise ValueError("Extension repeats an old/new document (including validation)")
                    if not db.execute("SELECT 1 FROM main.documents WHERE digest=?", (key,)).fetchone():
                        raise ValueError("New document absent from the committed dedup index")
                    seen.add(key)
        count = db.execute("SELECT (SELECT COUNT(*) FROM main.documents) - (SELECT COUNT(*) FROM ancestor.documents)").fetchone()[0]
        if count != len(seen):
            raise ValueError("Incremental provenance and global document index differ")
    return [numbers[0], numbers[-1]]


def extend_dataset_resume(checkpoint, parent_fingerprint, parent_root, new_root,
                          train_dataset, val_dataset, tokenizer_dir, context, seed):
    """Prove the consumed segment, then return new metadata; never reset ML state.

    The first parent checkpoint has only a fingerprint. Reconstruct it from the
    exact parent artifacts and original config, including its original validation.
    Only a completely consumed segment may hand off to the next incremental root.
    """
    if context.get("stage") != "foundation":
        raise ValueError("Dataset extension supports foundation only; SFT remains blocked")
    r = checkpoint.get("resume_state") or {}
    if r.get("fingerprint") != parent_fingerprint:
        raise ValueError("Parent fingerprint mismatch; already extended checkpoints must use plain --resume")
    if (not {"python", "torch_cpu", "torch_cuda"} <= (checkpoint.get("rng_state") or {}).keys()
            or "scaler_state_dict" not in checkpoint):
        raise ValueError("Dataset extension requires saved RNG and scaler history")
    if r.get("legacy_sample_cursor") is not None:
        raise ValueError("Legacy cursor migrations are not eligible for dataset extension")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Extension requires an explicit seed in [0, 2**63)")
    budget = context.get("input_token_budget")
    if type(budget) is not int or budget <= r.get("input_tokens_seen", 0):
        raise ValueError("Extension requires remaining cumulative real-input budget")
    parent, parent_hash, parent_db = _preparation(parent_root)
    new, new_hash, new_db = _preparation(new_root)
    cfg = context["config"]["model"]
    descriptors = select_production_shards(parent_root, "foundation", "train", tokenizer_dir,
                                          cfg["max_position_embeddings"], cfg["vocab_size"])
    old = LazyShardDataset(parent_root, seq_length=cfg["max_position_embeddings"],
                           vocab_size=cfg["vocab_size"], descriptors=descriptors)
    records = {"train": old.audit_records, "validation": val_dataset.audit_records}
    actual = dataset_fingerprint(records, Path(tokenizer_dir) / "tokenizer.json", context, r)
    training = context["config"]["training"]
    seqs_per_step = training.get("batch_size_per_device", 16) * training.get("gradient_accumulation_steps", 16)
    validate_resume_metadata(checkpoint, actual, seqs_per_step, budget)
    if (type(checkpoint.get("step")) is not int or checkpoint["step"] < r["completed_updates"]
            or r["completed_updates"] != r["optimizer_step_count"]):
        raise ValueError("Parent schedule/applied-update/optimizer counters disagree")
    if r.get("order_version") not in ("legacy-filtered-v1", ORDER_VERSION):
        raise ValueError("Unknown parent permutation identity")
    cursor, start_tokens = extension_position(old, r)
    if cursor != len(old) or r["input_tokens_seen"] != start_tokens + sum(old.input_tokens(i) for i in range(len(old))):
        raise ValueError("Parent segment must be fully consumed with verified real-input accounting")
    # Both datasets must be bound to the exact preparation receipts just verified.
    if any(record.get("identity", {}).get("preparation_manifest_sha256") != new_hash
           for record in train_dataset.audit_records):
        raise ValueError("New training rows are not from the supplied incremental manifest")
    window_range = _verify_chain(new_root, parent, new, parent_hash, parent_db, new_db)
    consumed = {"train_records_sha256": _digest(old.audit_records), "rows": len(old),
                "input_tokens": r["input_tokens_seen"] - start_tokens,
                "preparation_manifest_sha256": parent_hash,
                "document_index_sha256": parent["state_sha256"]}
    chain = r.get("lineage_chain", [])
    entry = {"parent_fingerprint": parent_fingerprint, "parent_lineage_sha256": _digest(chain),
             "consumed_old_set_identity": consumed, "start_sample_cursor": r["sample_cursor"],
             "start_input_tokens": r["input_tokens_seen"], "schedule_step": checkpoint["step"],
             "new_preparation_manifest_sha256": new_hash, "new_window_range": window_range,
             "validation_identity": _digest(val_dataset.audit_records), "order_seed": seed}
    result = copy.deepcopy(r)
    result.pop("legacy_sample_cursor", None)
    result.update(extension=entry, identity_context=copy.deepcopy(context),
                  parent_fingerprint=parent_fingerprint, consumed_old_set_identity=consumed,
                  order_version=ORDER_VERSION, order_seed=seed)
    records = {"train": train_dataset.audit_records, "validation": val_dataset.audit_records}
    fingerprint = manifest_fingerprint(records, Path(tokenizer_dir) / "tokenizer.json",
                                       {**context, "extension": entry})
    result.update(fingerprint=fingerprint, new_fingerprint=fingerprint,
                  lineage_chain=chain + [{"transition": entry, "new_fingerprint": fingerprint}])
    return result


def commit_dataset_extension(path, checkpoint, resume_state):
    """Atomic activation before update one: crash => old parent OR one handoff.

    Keep all model/optimizer/scaler/RNG tensors byte-for-byte as loaded. No step
    checkpoint rotation, uploads or RNG draws are involved in this metadata commit.
    """
    path = Path(path)
    temporary = path.with_name(path.name + ".extend.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save({**checkpoint, "resume_state": resume_state}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
