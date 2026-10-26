"""Bounded, public-source CPU pilot. This module never loads or trains a model."""
import argparse
import fnmatch
import hashlib
import importlib.metadata
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import yaml

from data.deduplicate import deduplicate_dataset
from data.filter_quality import filter_code_quality
from data.tokenize_dataset import build_tokenized_dataset
from data.train_tokenizer import train_custom_bpe_tokenizer


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_files(spec):
    from huggingface_hub import HfApi
    revision = spec["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Every source must have an immutable commit revision")
    info = HfApi(token=False).dataset_info(spec["repo"], revision=revision)
    if info.sha != revision or info.gated:
        raise ValueError(f"Pilot requires a pinned, ungated public source: {spec['repo']}")
    files = sorted(s.rfilename for s in info.siblings or []
                   if fnmatch.fnmatchcase(s.rfilename, spec["file_pattern"]))
    if not files:
        raise ValueError(f"No source files match {spec['repo']}/{spec['file_pattern']}")
    card = info.card_data.to_dict() if info.card_data is not None else {}
    return files, card.get("license")


def stream_file(spec, filename):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(spec["repo"], filename, repo_type="dataset", revision=spec["revision"])
    options = {}
    if spec.get("loader", "parquet") == "parquet":
        from pyarrow.dataset import ParquetFragmentScanOptions
        # A bounded consumer may stop within a row group. Do not leave native
        # Arrow workers reading Python-backed HF files during interpreter exit.
        options["fragment_scan_options"] = ParquetFragmentScanOptions(pre_buffer=False)
    # Built-in JSON/Parquet readers only: never execute a remote dataset script.
    return load_dataset(spec.get("loader", "parquet"), data_files={"train": [url]},
                        split="train", streaming=True, token=False, **options)


def lexical_windows(text, width, minimum_distinct):
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    for index in range(max(0, len(tokens) - width + 1)):
        window = tokens[index:index + width]
        if len(set(window)) >= minimum_distinct:
            yield hashlib.sha256(" ".join(window).encode("utf-8")).digest()


def build_benchmark_index(policy):
    index = set()
    records = []
    for spec in policy["sources"]:
        files, _ = source_files(spec)
        count = 0
        for filename in files:
            for row in stream_file(spec, filename):
                count += 1
                for field in spec["fields"]:
                    value = row.get(field, "")
                    text = "\n".join(str(item) for item in value) if isinstance(value, list) else str(value)
                    index.update(lexical_windows(text, policy["window_tokens"],
                                                 policy["minimum_distinct_tokens"]))
        records.append({"repo": spec["repo"], "revision": spec["revision"],
                        "files": files, "rows": count})
    if not index:
        raise ValueError("Benchmark exclusion index is empty")
    return index, records


def collect_source(spec, raw_dir, exclusion, policy, max_document_chars):
    files, corpus_license = source_files(spec)
    corpus_licenses = corpus_license if isinstance(corpus_license, list) else [corpus_license]
    allowed = set(spec["allowed_licenses"])
    if not spec.get("license_field") and not any(value in allowed for value in corpus_licenses):
        raise ValueError(f"Unapproved corpus license for {spec['repo']}")
    counts = Counter()
    licenses = Counter()
    used_files = []
    next_cursor = None
    path = raw_dir / (spec["name"] + "_raw.jsonl")
    with path.open("x", encoding="utf-8") as output:
        for filename in files:
            used_files.append(filename)
            for row_index, row in enumerate(stream_file(spec, filename)):
                counts["rows_examined"] += 1
                text = row.get(spec["text_field"], "")
                if not isinstance(text, str) or len(text) < 30 or len(text) > max_document_chars:
                    counts["size_rejected"] += 1
                    continue
                license_value = row.get(spec["license_field"]) if spec.get("license_field") else corpus_licenses[0]
                if license_value not in allowed:
                    counts["license_rejected"] += 1
                    continue
                if any(window in exclusion for window in lexical_windows(
                        text, policy["window_tokens"], policy["minimum_distinct_tokens"])):
                    counts["benchmark_match_rejected"] += 1
                    continue
                record = {
                    "text": text, "source": spec["name"], "repo": spec["repo"],
                    "revision": spec["revision"], "file": filename, "row": row_index,
                    "license": license_value, "repo_name": row.get("repo_name"),
                    "path": row.get("path"), "url": row.get("url"),
                    "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["documents"] += 1
                counts["written_chars"] += len(text)
                licenses[license_value] += 1
                next_cursor = {"file": filename, "next_row": row_index + 1}
                if counts["written_chars"] >= spec["max_chars"]:
                    break
            if counts["written_chars"] >= spec["max_chars"]:
                break
    if counts["written_chars"] < spec["max_chars"]:
        raise ValueError(f"Source exhausted before the requested pilot cap: {spec['name']}")
    return {"repo": spec["repo"], "revision": spec["revision"], "files": used_files,
            "corpus_license": corpus_license, "retained_license_counts": dict(licenses),
            "counts": dict(counts), "next_cursor": next_cursor, "raw_sha256": file_sha256(path)}


def partition_documents(dedup_dir, train_dir, validation_dir, fraction, seed):
    if not 0 < fraction < 0.5:
        raise ValueError("Validation fraction must be between zero and one half")
    train_dir.mkdir()
    validation_dir.mkdir()
    counts = {"train": Counter(), "validation": Counter()}
    split_hashes = {"train": set(), "validation": set()}
    threshold = int(fraction * (1 << 64))
    for source in sorted(dedup_dir.glob("*_dedup.jsonl")):
        name = source.name.removesuffix("_dedup.jsonl")
        with source.open(encoding="utf-8") as incoming, \
                (train_dir / source.name).open("x", encoding="utf-8") as train, \
                (validation_dir / source.name).open("x", encoding="utf-8") as validation:
            for line in incoming:
                record = json.loads(line)
                normalized = " ".join(record["text"].split())
                digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                value = hashlib.sha256(f"{seed}:{digest}".encode()).digest()
                split = "validation" if int.from_bytes(value[:8], "big") < threshold else "train"
                (validation if split == "validation" else train).write(line)
                counts[split][name] += 1
                split_hashes[split].add(digest)
    if not all(split_hashes.values()) or split_hashes["train"] & split_hashes["validation"]:
        raise ValueError("Document split is empty or has normalized exact overlap")
    return {key: dict(value) for key, value in counts.items()}


def audit_packed(directory, manifest, vocab_size, seq_len):
    import torch
    actual_valid = actual_rows = 0
    for item in manifest["shards"]:
        shard = directory / item["file"]
        sidecar = directory / item["valid_file"]
        if file_sha256(shard) != item["sha256"] or file_sha256(sidecar) != item["valid_sha256"]:
            raise ValueError("Packed artifact checksum mismatch")
        tensor = torch.load(shard, map_location="cpu", weights_only=True)
        lengths = torch.load(sidecar, map_location="cpu", weights_only=True)["valid_len"]
        if (tensor.dtype != torch.uint16 or tensor.ndim != 2 or tensor.shape[1] != seq_len
                or lengths.shape != (tensor.shape[0],) or int(lengths.min()) < 2
                or int(lengths.max()) > seq_len or int(tensor.to(torch.int32).max()) >= vocab_size):
            raise ValueError("Invalid shard or valid-length layout")
        actual_rows += tensor.shape[0]
        actual_valid += int(lengths.sum())
    if (actual_rows != manifest["sequences"] or actual_valid != manifest["valid_tokens"]
            or actual_valid + manifest["discarded_tokens"] != manifest["total_tokens"]):
        raise ValueError("Packed token accounting mismatch")


def prepare(config_path, output):
    start = time.time()
    config_path, output = Path(config_path), Path(output)
    if output.exists():
        raise FileExistsError("Fresh preparation refuses an existing output directory")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    raw, filtered, dedup = (output / name for name in ("raw", "filtered", "dedup"))
    raw.mkdir()
    exclusion, benchmarks = build_benchmark_index(config["benchmark_exclusion"])
    collection = {}
    for spec in config["sources"]:
        print(f"Collecting pinned public source {spec['name']}", flush=True)
        collection[spec["name"]] = collect_source(
            spec, raw, exclusion, config["benchmark_exclusion"], config["max_document_chars"])
    write_json(output / "collection.json", collection)
    filter_code_quality(str(raw), str(filtered))
    dedup_stats = deduplicate_dataset(str(filtered), str(dedup))
    train_text, validation_text = output / "train_text", output / "validation_text"
    split_counts = partition_documents(dedup, train_text, validation_text,
                                       config["validation_fraction"], config["seed"])
    tokenizer = output / "tokenizer"
    train_custom_bpe_tokenizer(str(train_text), str(tokenizer), config["vocab_size"])
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer)
    required = ["<|endoftext|>", "<|unk|>", "<|pad|>", "<|fim_prefix|>", "<|fim_middle|>",
                "<|fim_suffix|>", "<|tool_call|>", "<|tool_result|>", "<|thinking|>", "<|json_start|>", "<|json_end|>"]
    if (len(tok) != config["vocab_size"] or tok.unk_token_id == tok.pad_token_id
            or any(len(tok.encode(token, add_special_tokens=False)) != 1 for token in required)
            or len({tok.convert_tokens_to_ids(token) for token in required}) != len(required)):
        raise ValueError("Candidate tokenizer identity/special-token gate failed")
    manifests = {}
    for split, text_dir, target in (("train", train_text, output / "tokenized"),
                                     ("validation", validation_text, output / "validation")):
        manifests[split] = build_tokenized_dataset(str(text_dir), str(tokenizer), str(target), config["seq_length"])
        audit_packed(target, manifests[split], config["vocab_size"], config["seq_length"])
    if (manifests["train"]["sequences"] < config["min_train_sequences"]
            or manifests["validation"]["sequences"] < config["min_validation_sequences"]):
        raise ValueError("Pilot retained too little training or validation data")
    if manifests["train"]["tokenizer"]["sha256"] != manifests["validation"]["tokenizer"]["sha256"]:
        raise ValueError("Train and validation tokenizers differ")
    # Keep admitted, attributed text and packed data. Raw/filter/dedup intermediates
    # are removed only after both packed splits pass independent readback.
    write_json(output / "dedup_stats.json", dedup_stats)
    for directory in (raw, filtered, dedup):
        shutil.rmtree(directory)
    shutil.copyfile(config_path, output / "preparation-config.yaml")
    summary = {
        "run_id": config["run_id"], "status": "cpu-data-pilot-verified",
        "model_training_started": False, "production_tokenizer_promoted": False,
        "config_sha256": file_sha256(config_path), "collection": collection,
        "benchmark_exclusion": {"policy": config["benchmark_exclusion"], "sources_read": benchmarks,
                                "window_count": len(exclusion), "complete_decontamination_claim": False},
        "deduplication": dedup_stats, "document_splits": split_counts,
        "packed": manifests, "elapsed_seconds": round(time.time() - start, 2),
        "software": {name: importlib.metadata.version(name) for name in
                     ("torch", "datasets", "datasketch", "transformers", "tokenizers")},
    }
    write_json(output / "preparation-manifest.json", summary)
    print(json.dumps({"status": summary["status"], "train_valid_tokens": manifests["train"]["valid_tokens"],
                      "validation_valid_tokens": manifests["validation"]["valid_tokens"],
                      "model_training_started": False}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.config, args.output)


if __name__ == "__main__":
    main()
