"""Fail-closed checks for CodeForge's single-GPU, immutable Stage A resume."""
import hashlib
import json
from pathlib import Path
import torch


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit_shards(paths, seq_length, vocab_size, allow_partial=False):
    records = []
    expected_rows = None
    for path in paths:
        t = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(t, torch.Tensor) or t.ndim != 2:
            raise ValueError(f'Invalid shard tensor: {Path(path).name}')
        if t.dtype != torch.uint16 or t.shape[1] != seq_length or t.shape[0] < 1:
            raise ValueError(f'Invalid shard dtype/shape: {Path(path).name}')
        rows = int(t.shape[0])
        if expected_rows is None:
            expected_rows = rows
        if rows != expected_rows and not allow_partial:
            raise ValueError(f'Unequal shard rows: {Path(path).name}. Legacy permutation cannot be preserved.')
        if int(t.to(torch.int32).max()) >= vocab_size:
            raise ValueError(f'Out-of-vocabulary ID: {Path(path).name}')
        sidecars = {}
        for prefix in ('valid_', 'loss_'):
            side = Path(path).with_name(Path(path).name.replace('shard_', prefix, 1))
            sidecars[prefix.rstrip('_')] = (
                {'bytes': side.stat().st_size, 'sha256': file_sha256(side)}
                if side != Path(path) and side.is_file() else None)
        records.append({'name':Path(path).name, 'rows':rows, 'seq_length':seq_length,
                        'bytes':Path(path).stat().st_size, 'sha256':file_sha256(path),
                        'sidecars': sidecars})
        del t  # Do not retain the preceding shard while loading the next one.
    if not records:
        raise ValueError('No shards')
    return records


def manifest_fingerprint(records, tokenizer_path, context=None):
    payload = {'shards':records, 'tokenizer_sha256':file_sha256(tokenizer_path),
               'context': context}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def optimizer_step_counts(optimizer_state):
    values = []
    for state in optimizer_state['state'].values():
        if 'step' in state:
            v = state['step']
            values.append(int(v.item() if isinstance(v,torch.Tensor) else v))
    if not values or len(set(values)) != 1:
        raise ValueError('Optimizer counters missing or inconsistent')
    return values[0]


def validate_resume_metadata(state, fingerprint, seqs_per_step, token_budget=None):
    r = state.get('resume_state')
    if r is None:
        raise ValueError('Legacy checkpoint requires explicit audited migration metadata')
    if r.get('schema_version') != 2:
        raise ValueError('Checkpoint requires audited migration to real-token accounting schema 2')
    if r.get('fingerprint') != fingerprint:
        raise ValueError('Checkpoint data/tokenizer/config identity mismatch')
    if r.get('seqs_per_step') != seqs_per_step:
        raise ValueError('Checkpoint batch geometry mismatch')
    for field in ('sample_cursor', 'completed_updates', 'optimizer_step_count',
                  'input_tokens_seen'):
        if type(r.get(field)) is not int or r[field] < 0:
            raise ValueError('Checkpoint counters invalid')
    if token_budget is not None and r['input_tokens_seen'] > token_budget:
        raise ValueError('Checkpoint real-token counter exceeds budget')
    if r.get('optimizer_step_count') != optimizer_step_counts(state['optimizer_state_dict']):
        raise ValueError('Checkpoint optimizer metadata mismatch')
    return r


def reconcile_legacy_cursor(reported_step, prior_start_step, session_prediction_tokens,
                            seqs_per_step, seq_length):
    """Recover cursor from a matching clean-session log, not optimizer counters."""
    per_batch = seqs_per_step * (seq_length - 1)
    batches, remainder = divmod(session_prediction_tokens, per_batch)
    if remainder or reported_step != prior_start_step + batches + 1:
        raise ValueError('Legacy session log does not reconcile with one-ahead wall stop')
    schedule_step = prior_start_step + batches
    return schedule_step, schedule_step * seqs_per_step


def migrated_permutation(rows, legacy_rows=1000, seed=42):
    """Filter nonexistent legacy rows, preserving the old order of real rows."""
    if any(r < 1 or r > legacy_rows for r in rows):
        raise ValueError('Unsupported legacy row geometry')
    lookup = torch.full((len(rows)*legacy_rows,), -1, dtype=torch.int64)
    offset = 0
    for i,r in enumerate(rows):
        lookup[i*legacy_rows:i*legacy_rows+r] = torch.arange(offset,offset+r)
        offset += r
    old = torch.randperm(len(lookup), generator=torch.Generator().manual_seed(seed))
    mapped = lookup[old]
    valid = mapped >= 0
    return mapped[valid].tolist(), valid
