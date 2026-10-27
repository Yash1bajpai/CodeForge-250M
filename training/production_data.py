"""Validated production manifests and lazy, bounded-memory shard consumption."""

from bisect import bisect_right
import hashlib
import json
import operator
import os
from pathlib import Path, PurePosixPath

import torch
from torch.utils.data import Dataset

from training.resume_safety import audit_shards, file_sha256


_ACCOUNTING = ("valid_tokens", "padding_tokens", "loss_tokens")
_PAYLOAD_FIELDS = ("file", "valid_file", "loss_file")


def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read production manifest/receipt: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Production manifest/receipt must be an object: {path}")
    return value


def _safe_path(root, relative):
    """Resolve only explicit, portable manifest paths contained in their root."""
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))
            or PurePosixPath(relative).is_absolute()):
        raise ValueError(f"Unsafe production relative path: {relative!r}")
    path = root.joinpath(*PurePosixPath(relative).parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Production path escapes root: {relative!r}")
    return path


def _checked_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Invalid production checksum: {path}")
    try:
        actual = file_sha256(path)
    except OSError as error:
        raise ValueError(f"Missing/unreadable production file: {path}") from error
    if actual != expected:
        raise ValueError(f"Production checksum mismatch: {path}")
    return actual


def _tokenizer_identity(directory):
    """Match data.prepare_production.tokenizer_identity without producer imports."""
    if not directory.is_dir():
        raise ValueError(f"Tokenizer directory is missing: {directory}")
    files = {path.name: file_sha256(path) for path in sorted(directory.iterdir())
             if path.is_file()}
    if not files:
        raise ValueError(f"Tokenizer directory is empty: {directory}")
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"files": files, "sha256": digest}


def _count(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid production {label}: {value!r}")
    return value


def _load_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"Invalid production tensor payload: {path}") from error


def _validate_payloads(paths, item, seq_length, vocab_size):
    """Audit one shard at a time; retain no corpus tensors or loss masks."""
    ids = _load_payload(paths["file"])
    valid_payload = _load_payload(paths["valid_file"])
    loss = _load_payload(paths["loss_file"])
    valid = valid_payload.get("valid_len") if isinstance(valid_payload, dict) else None
    rows = _count(item.get("rows"), "shard rows", minimum=1)
    if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.uint16
            or ids.ndim != 2 or tuple(ids.shape) != (rows, seq_length)
            or not isinstance(valid, torch.Tensor) or valid.dtype != torch.int32
            or valid.ndim != 1 or len(valid) != rows
            or not isinstance(loss, torch.Tensor) or loss.dtype != torch.uint8
            or tuple(loss.shape) != (rows, seq_length)):
        raise ValueError(f"Invalid production token/length/loss layout: {paths['file']}")
    if int(valid.min()) < 2 or int(valid.max()) > seq_length:
        raise ValueError(f"Invalid production valid lengths: {paths['valid_file']}")
    if int(ids.to(torch.int32).max()) >= vocab_size:
        raise ValueError(f"Out-of-vocabulary production token ID: {paths['file']}")
    if int(loss.max()) > 1 or bool(loss[:, 0].any()):
        raise ValueError(f"Invalid production loss mask: {paths['loss_file']}")
    for row, length in enumerate(valid.tolist()):
        if bool(loss[row, length:].any()):
            raise ValueError(f"Production padding has supervised targets: {paths['loss_file']}")
        if item["stage"] == "foundation" and not bool(loss[row, 1:length].all()):
            raise ValueError(f"Foundation loss mask must equal valid causal targets: {paths['loss_file']}")
    measured = {"valid_tokens": int(valid.sum()), "loss_tokens": int(loss.sum()),
                "padding_tokens": ids.numel() - int(valid.sum())}
    for field, actual in measured.items():
        if _count(item.get(field), field) != actual:
            raise ValueError(f"Production shard accounting mismatch ({field}): {paths['file']}")
    return measured


def _validate_window(window, relative, manifest_hash, tokenizer_hash, seq_length,
                     vocab_size, stage, split, seen_paths, number=None):
    manifest_path = _safe_path(window, "manifest.json")
    actual_hash = _checked_hash(manifest_path, manifest_hash) if manifest_hash else file_sha256(manifest_path)
    manifest = _read_json(manifest_path)
    window_number = _count(manifest.get("window"), "window number", minimum=1)
    if number is not None and window_number != number:
        raise ValueError("Committed production window number does not match its manifest")
    if _count(manifest.get("seq_len"), "sequence length", minimum=2) != seq_length:
        raise ValueError("Production sequence length mismatch")
    if manifest.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("Production tokenizer identity mismatch")
    _checked_hash(_safe_path(window, "documents.jsonl"), manifest.get("documents_sha256"))
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("Production window shards must be a list")
    totals = dict.fromkeys(_ACCOUNTING, 0)
    descriptors = []
    for item in shards:
        if not isinstance(item, dict):
            raise ValueError("Production shard entry must be an object")
        if any(not isinstance(item.get(field), str) or not item[field]
               for field in ("source", "stage", "split")):
            raise ValueError("Production shard source/stage/split is missing")
        if item["split"] not in ("train", "validation"):
            raise ValueError("Invalid production shard split")
        paths = {}
        for field in _PAYLOAD_FIELDS:
            path = _safe_path(window, item.get(field))
            if path in seen_paths:
                raise ValueError(f"Duplicate production shard/sidecar path: {path}")
            seen_paths.add(path)
            _checked_hash(path, item.get(field + "_sha256"))
            paths[field] = path
        measured = _validate_payloads(paths, item, seq_length, vocab_size)
        for field in _ACCOUNTING:
            totals[field] += measured[field]
        if item["stage"] == stage and item["split"] == split:
            identity = {"window": relative, "window_number": window_number,
                        "manifest_sha256": actual_hash,
                        **{field: item[field] for field in ("source", "stage", "split")},
                        **{field: item[field] for field in _PAYLOAD_FIELDS},
                        **{field + "_sha256": item[field + "_sha256"] for field in _PAYLOAD_FIELDS}}
            descriptors.append({"path": str(paths["file"]),
                                "valid_path": str(paths["valid_file"]),
                                "loss_path": str(paths["loss_file"]), "identity": identity})
    for field, total in totals.items():
        if _count(manifest.get(field), field) != total:
            raise ValueError(f"Production window accounting mismatch ({field}): {window}")
    return descriptors


def select_production_shards(root, stage, split, tokenizer_dir, seq_length, vocab_size):
    """Select only pinned foundation shards, never discovering or following parents.

    A preparation root supplies its owned-window manifest pins and tokenizer receipt.
    A standalone committed window supplies its own manifest and tokenizer hash.
    Every committed shard is audited, including nonselected splits, before descriptors
    are returned. Instruction consumption is intentionally unsupported.
    """
    if stage != "foundation":
        raise ValueError("Instruction production consumption is unsupported; foundation only")
    split = "validation" if split == "val" else split
    if split not in ("train", "validation"):
        raise ValueError(f"Unsupported production split: {split!r}")
    _count(seq_length, "sequence length", minimum=2)
    _count(vocab_size, "vocabulary size", minimum=1)
    root = Path(root).resolve()
    tokenizer = _tokenizer_identity(Path(tokenizer_dir).resolve())
    preparation_path = _safe_path(root, "preparation-manifest.json")
    windows = []
    preparation_hash = None
    if preparation_path.is_file():
        preparation = _read_json(preparation_path)
        if preparation.get("format_version") != 2:
            raise ValueError("Unsupported production preparation manifest format")
        preparation_hash = file_sha256(preparation_path)
        receipt = _read_json(_safe_path(root, "tokenizer-identity.json"))
        if receipt != tokenizer or preparation.get("tokenizer") != tokenizer:
            raise ValueError("Production tokenizer identity/receipt mismatch")
        owned = preparation.get("owned_windows")
        if not isinstance(owned, list):
            raise ValueError("Production owned_windows must be a list")
        seen_numbers = set()
        for item in owned:
            if not isinstance(item, dict):
                raise ValueError("Invalid owned production window")
            number = _count(item.get("number"), "owned window number", minimum=1)
            if number in seen_numbers:
                raise ValueError("Duplicate owned production window")
            seen_numbers.add(number)
            relative = f"windows/window-{number:06d}"
            window = _safe_path(root, relative)
            manifest_hash = item.get("manifest_sha256")
            if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
                raise ValueError("Owned production window lacks a valid manifest pin")
            windows.append((window, relative, manifest_hash, number))
    elif root.name.endswith(".part"):
        raise ValueError("Uncommitted production window cannot be consumed")
    elif _safe_path(root, "manifest.json").is_file():
        windows.append((root, ".", None, None))
    else:
        raise ValueError("Expected a production preparation root or committed window manifest")
    descriptors = []
    seen_paths = set()
    for window, relative, manifest_hash, number in windows:
        descriptors.extend(_validate_window(window, relative, manifest_hash, tokenizer["sha256"],
                                            seq_length, vocab_size, stage, split, seen_paths, number))
    if preparation_hash is not None:
        for descriptor in descriptors:
            descriptor["identity"]["preparation_manifest_sha256"] = preparation_hash
    if not descriptors:
        raise ValueError(f"No production shards selected for {stage}/{split}")
    return descriptors



class LazyShardDataset(Dataset):
    """Validate once, retain row lengths, and load at most one token shard.

    Loss masks are accepted only when they describe every real causal target.
    This loader intentionally cannot consume instruction/SFT loss masking.
    """

    def __init__(self, tokenized_dir, seq_length=2048, shard_files=None,
                 vocab_size=32000, descriptors=None, return_input_tokens=False):
        self.seq_length = operator.index(seq_length)
        if self.seq_length < 2:
            raise ValueError("seq_length must be at least 2")
        self.return_input_tokens = return_input_tokens
        if descriptors is not None and shard_files is not None:
            raise ValueError("Pass descriptors or shard_files, not both")
        if descriptors is not None:
            descriptors = list(descriptors)
            paths = [item["path"] for item in descriptors]
        elif shard_files is not None:
            paths = list(shard_files)
        else:
            paths = sorted(Path(tokenized_dir).glob("shard_*.pt"))
        self.shard_files = [str(Path(path).resolve()) for path in paths]
        if not self.shard_files:
            raise ValueError("No shards selected")
        if len(set(self.shard_files)) != len(self.shard_files):
            raise ValueError("Duplicate shard paths")
        self.audit_records = audit_shards(
            self.shard_files, self.seq_length, vocab_size, allow_partial=True)
        self.rows = [record["rows"] for record in self.audit_records]
        self.offsets = [0]
        self._valid_lengths = []
        for index, (path, record) in enumerate(zip(self.shard_files, self.audit_records)):
            path = Path(path)
            if descriptors is None:
                valid_path = self._legacy_sidecar(path, "valid_")
                loss_path = self._legacy_sidecar(path, "loss_")
            else:
                descriptor = descriptors[index]
                valid_path = Path(descriptor["valid_path"])
                loss_path = Path(descriptor["loss_path"])
                identity = descriptor.get("identity")
                if not isinstance(identity, dict) or not identity:
                    raise ValueError("Production descriptor requires a stable identity")
                record["identity"] = json.loads(json.dumps(identity, sort_keys=True))
            lengths = self._read_lengths(valid_path, record["rows"])
            self._validate_loss(loss_path, lengths)
            record["sidecars"] = {
                "valid": self._sidecar_identity(valid_path),
                "loss": self._sidecar_identity(loss_path),
            }
            self._valid_lengths.append(lengths)
            self.offsets.append(self.offsets[-1] + record["rows"])
        self._cached_shard = None
        self._cached_shard_index = None
        self._cache_pid = os.getpid()

    @staticmethod
    def _legacy_sidecar(path, prefix):
        if not path.name.startswith("shard_"):
            return None
        candidate = path.with_name(prefix + path.name[len("shard_"):])
        return candidate if candidate.exists() else None

    @staticmethod
    def _sidecar_identity(path):
        if path is None:
            return None
        return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}

    def _read_lengths(self, path, rows):
        if path is None:
            return torch.full((rows,), self.seq_length, dtype=torch.int32)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        lengths = payload.get("valid_len") if isinstance(payload, dict) else None
        if (not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.int32
                or lengths.ndim != 1 or lengths.shape[0] != rows):
            raise ValueError(f"Invalid valid_len sidecar dtype/shape: {path.name}")
        if int(lengths.min()) < 2 or int(lengths.max()) > self.seq_length:
            raise ValueError(f"Invalid valid_len sidecar range: {path.name}")
        # Own only a compact vector, never a view backed by a larger tensor.
        return lengths.clone()

    def _validate_loss(self, path, lengths):
        if path is None:
            return
        mask = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.uint8
                or mask.ndim != 2 or tuple(mask.shape) != (len(lengths), self.seq_length)):
            raise ValueError(f"Invalid loss mask dtype/shape: {path.name}")
        for row, length in zip(mask, lengths):
            length = int(length)
            if (int(row[0]) != 0 or not bool(torch.all(row[1:length] == 1))
                    or bool(torch.any(row[length:] != 0))):
                raise ValueError(
                    f"Loss mask must equal valid causal targets; instruction/SFT masks "
                    f"are unsupported: {path.name}")

    def __len__(self):
        return self.offsets[-1]

    def _locate(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Shard dataset index out of range")
        shard = bisect_right(self.offsets, index) - 1
        return shard, index - self.offsets[shard]

    def input_tokens(self, index):
        """Count real x tokens without loading a token shard."""
        shard, row = self._locate(index)
        return min(int(self._valid_lengths[shard][row]), self.seq_length - 1)

    def __getitem__(self, index):
        shard, row = self._locate(index)
        pid = os.getpid()
        if self._cache_pid != pid or self._cached_shard_index != shard:
            # Release before loading: assignment alone transiently retains two shards.
            self._cached_shard = None
            self._cached_shard_index = None
            self._cached_shard = torch.load(
                self.shard_files[shard], map_location="cpu", weights_only=True)
            self._cached_shard_index = shard
            self._cache_pid = pid
        sequence = self._cached_shard[row].long()
        x = sequence[:-1]
        y = sequence[1:].clone()
        valid = int(self._valid_lengths[shard][row])
        y[valid - 1:] = -100
        if self.return_input_tokens:
            return x, y, min(valid, self.seq_length - 1)
        return x, y

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cached_shard"] = None
        state["_cached_shard_index"] = None
        state["_cache_pid"] = None
        return state
