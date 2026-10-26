import glob
import hashlib
import json
import os
import random

import torch
from transformers import PreTrainedTokenizerFast

SEED = 42
SHARD_SIZE = 1000

# FIM is applied ONLY to plain source-code corpora. Prose (fineweb-edu), chat /
# instruction data (evol-codealpaca), tool-call data (glaive) and commitpackft
# (already carries its own prefix/middle format) must never get FIM tokens:
# applying FIM there taught the model to emit FIM tokens on normal prompts.
FIM_SOURCES = ("starcoder-python", "codeparrot-clean")
FIM_TOKENS = ("<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>")

def source_name(file_path: str) -> str:
    """'data/dedup/starcoder-python_dedup.jsonl' -> 'starcoder-python'"""
    return os.path.basename(file_path).split("_")[0]

def fim_allowed(source: str) -> bool:
    return source in FIM_SOURCES

def apply_fim_transformation(code: str, fim_rate: float = 0.50, rng=None) -> str:
    """Reorder complete, unchanged lines of raw code; never wrap existing FIM."""
    if any(t in code for t in FIM_TOKENS):
        return code
    rng = rng if rng is not None else random.Random()
    if rng.random() > fim_rate or len(code) < 50:
        return code
    lines = code.splitlines(keepends=True)
    if len(lines) < 3:
        return code
    split1 = rng.randint(1, len(lines) - 2)
    split2 = rng.randint(split1 + 1, len(lines) - 1)
    prefix = "".join(lines[:split1])
    middle = "".join(lines[split1:split2])
    suffix = "".join(lines[split2:])
    return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}"

def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_tokenized_dataset(dedup_dir: str = "data/dedup", tokenizer_dir: str = "data/tokenizer", output_dir: str = "data/tokenized", seq_len: int = 2048):
    """Pack one already-split corpus, recording every real and padding token.

    Memory is bounded by one encoded document, one row and one 1,000-row shard.
    ``valid_*.pt`` contains real row lengths for the trainer's target loss mask.
    A final single token has no next-token target and is explicitly discarded.
    Output must be empty; a failed or completed build is never overwritten.
    """
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2 to provide next-token targets")
    if not os.path.isdir(tokenizer_dir):
        raise FileNotFoundError(f"Tokenizer directory missing: {tokenizer_dir}")
    if not os.path.isdir(dedup_dir):
        raise FileNotFoundError(f"Deduplicated corpus directory missing: {dedup_dir}")
    files = sorted(glob.glob(os.path.join(dedup_dir, "*_dedup.jsonl")))
    if not files:
        raise FileNotFoundError(f"No *_dedup.jsonl files in {dedup_dir}")
    if os.path.exists(output_dir) and (
        not os.path.isdir(output_dir) or os.listdir(output_dir)
    ):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output_dir}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must define an EOS token")
    vocab = tokenizer.get_vocab()
    if not vocab or min(vocab.values()) < 0 or max(vocab.values()) > 65535:
        raise ValueError("Tokenizer vocabulary must fit uint16 token IDs")
    if not 0 <= eos_id <= 65535:
        raise ValueError("EOS token ID must fit uint16")
    tokenizer_files = {}
    for root, dirs, names in os.walk(tokenizer_dir):
        dirs.sort()
        for name in sorted(names):
            path = os.path.join(root, name)
            relative = os.path.relpath(path, tokenizer_dir).replace(os.sep, "/")
            tokenizer_files[relative] = _sha256(path)
    tokenizer_identity = {"files": tokenizer_files, "vocab": vocab}
    tokenizer_hash = hashlib.sha256(
        json.dumps(tokenizer_identity, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "seed": SEED,
        "seq_len": seq_len,
        "shard_size": SHARD_SIZE,
        "tokenizer": {
            "sha256": tokenizer_hash,
            "vocab_size": len(vocab),
            "eos_token_id": eos_id,
            "files": tokenizer_files,
        },
        "sources": {},
        "total_tokens": 0,
        "valid_tokens": 0,
        "padding_tokens": 0,
        "discarded_tokens": 0,
        "sequences": 0,
        "shards": [],
    }
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(SEED)
    pending = []
    rows = []
    valid_lengths = []

    def flush_shard():
        if not rows:
            return
        index = len(manifest["shards"]) + 1
        filename = f"shard_{index:04d}.pt"
        valid_filename = f"valid_{index:04d}.pt"
        path = os.path.join(output_dir, filename)
        valid_path = os.path.join(output_dir, valid_filename)
        # File handles avoid path-dependent archive names in torch serialization.
        with open(path, "xb") as handle:
            torch.save(torch.tensor(rows, dtype=torch.uint16), handle)
        with open(valid_path, "xb") as handle:
            torch.save({"valid_len": torch.tensor(valid_lengths, dtype=torch.int32)}, handle)
        valid_tokens = sum(valid_lengths)
        padding_tokens = len(rows) * seq_len - valid_tokens
        manifest["shards"].append({
            "file": filename,
            "sha256": _sha256(path),
            "valid_file": valid_filename,
            "valid_sha256": _sha256(valid_path),
            "rows": len(rows),
            "valid_tokens": valid_tokens,
            "padding_tokens": padding_tokens,
        })
        manifest["valid_tokens"] += valid_tokens
        manifest["padding_tokens"] += padding_tokens
        manifest["sequences"] += len(rows)
        rows.clear()
        valid_lengths.clear()

    def emit_row(row, valid_len):
        rows.append(row)
        valid_lengths.append(valid_len)
        if len(rows) == SHARD_SIZE:
            flush_shard()

    for file_path in files:
        source = source_name(file_path)
        stats = manifest["sources"].setdefault(source, {
            "documents": 0,
            "fim_documents": 0,
            "preformatted_documents": 0,
            "tokens": 0,
            "input_files": {},
        })
        stats["input_files"][os.path.basename(file_path)] = _sha256(file_path)
        with open(file_path, "r", encoding="utf-8", newline="") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    code = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    code = line.rstrip("\r\n").replace("\\n", "\n")
                preformatted = any(token in code for token in FIM_TOKENS)
                transformed = (
                    apply_fim_transformation(code, fim_rate=0.50, rng=rng)
                    if fim_allowed(source) else code
                )
                tokens = tokenizer.encode(transformed)
                tokens.append(eos_id)
                stats["documents"] += 1
                stats["fim_documents"] += int(transformed != code)
                stats["preformatted_documents"] += int(preformatted)
                stats["tokens"] += len(tokens)
                manifest["total_tokens"] += len(tokens)
                offset = 0
                while offset < len(tokens):
                    count = min(seq_len - len(pending), len(tokens) - offset)
                    pending.extend(tokens[offset:offset + count])
                    offset += count
                    if len(pending) == seq_len:
                        emit_row(pending, seq_len)
                        pending = []

    if len(pending) >= 2:
        valid_len = len(pending)
        pending.extend([eos_id] * (seq_len - valid_len))
        emit_row(pending, valid_len)
    elif pending:
        manifest["discarded_tokens"] = 1
    flush_shard()
    with open(os.path.join(output_dir, "manifest.json"), "x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"--> Packed {manifest['valid_tokens']:,} real tokens into "
        f"{manifest['sequences']:,} sequences in {output_dir}; "
        f"padding={manifest['padding_tokens']:,}, discarded={manifest['discarded_tokens']}"
    )
    return manifest

if __name__ == "__main__":
    build_tokenized_dataset()
