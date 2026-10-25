"""CodeForge CPU preflight (prep tooling; CPU only, no network, never executes model output).

Schema grounding: the checkpoint/resume invariants below mirror the audited training/train.py (full sha256
5e0ec2639492046f663e03ffed3aebfaea684f0fd7b07034562a569609e60fbb, recovered from the preserved project output) and its
training/resume_safety.py (identical to repo master except a trailing blank line): resume_state schema_version 1,
order_version 'legacy-filtered-v1'. Every check is a CONFIGURED expectation for that schema/lineage, not a universal law.
Status of each check on real data: see PREFLIGHT_STATUS.md (fixture-tested only unless stated).
Secret scanning is defense in depth, never a certification that a cell is clean."""
import ast, glob, hashlib, json, os, re

class PreflightError(Exception): pass

# ---------- helpers ----------
def find_exactly_one(pattern, what):
    hits = sorted(glob.glob(pattern, recursive=True))
    if len(hits) != 1:
        raise PreflightError(f"{what}: expected exactly 1 match for {pattern!r}, found {len(hits)}: {hits[:5]}")
    return hits[0]

def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block), b''): h.update(b)
    return h.hexdigest()

def check_sha(path, expected, what):
    """Computes first; the mismatch message carries the ACTUAL hash and size."""
    actual = sha256_file(path); size = os.path.getsize(path)
    if actual != expected:
        raise PreflightError(f"{what}: sha256 mismatch actual={actual} expected={expected} bytes={size}")
    return {'what': what, 'sha256': actual, 'bytes': size}

# ---------- data (same algorithm as resume_safety.audit_shards / manifest_fingerprint) ----------
def audit_shards(paths, seq_length, vocab_size, allow_partial=True):
    import torch
    recs = []
    for p in paths:
        t = torch.load(p, map_location='cpu', weights_only=True)
        if not isinstance(t, torch.Tensor) or t.ndim != 2 or t.dtype != torch.uint16 or t.shape[1] != seq_length or t.shape[0] < 1:
            raise PreflightError(f"bad shard layout {os.path.basename(p)} (expect 2-D uint16 x {seq_length})")
        if int(t.to(torch.int32).max()) >= vocab_size: raise PreflightError(f"out-of-vocab id in {os.path.basename(p)}")
        recs.append({'name': os.path.basename(p), 'rows': int(t.shape[0]), 'seq_length': seq_length,
                     'bytes': os.stat(p).st_size, 'sha256': sha256_file(p)})
    if not recs: raise PreflightError('no shards')
    if not allow_partial and len({r['rows'] for r in recs}) != 1: raise PreflightError('unequal shard rows')
    return recs

def manifest_fingerprint(records, tokenizer_path):
    return hashlib.sha256(json.dumps({'shards': records, 'tokenizer_sha256': sha256_file(tokenizer_path)}, sort_keys=True).encode()).hexdigest()

def check_sidecars(train_dir, records, seq_len):
    """Mirror of train.py LazyShardDataset sidecar validation: valid_XXXX.pt holds {'valid_len': tensor[rows]},
    min>=2, max<=seq_len, one sidecar per train shard with matching row count."""
    import torch
    for r in records:
        side = os.path.join(train_dir, r['name'].replace('shard_', 'valid_', 1))
        if not os.path.exists(side): raise PreflightError(f"missing sidecar for {r['name']}")
        d = torch.load(side, map_location='cpu', weights_only=True)
        if not isinstance(d, dict) or 'valid_len' not in d: raise PreflightError(f"sidecar {os.path.basename(side)} lacks 'valid_len'")
        vl = d['valid_len'].to(torch.int32)
        if vl.ndim != 1 or vl.shape[0] != r['rows'] or int(vl.min()) < 2 or int(vl.max()) > seq_len:
            raise PreflightError(f"bad valid_len in {os.path.basename(side)}")

def check_data_layout(train_dir, val_dir, tokenizer_path, expect, vocab=32000, seq_len=2048):
    """expect: dict with n_train_shards, n_valid_sidecars, min_val_shards, train_rows (optional), val_rows (optional),
    tokenizer_sha256 (optional), baseline_records (optional list from a prior record_baseline run)."""
    ts = sorted(glob.glob(os.path.join(train_dir, 'shard_*.pt'))); tv = sorted(glob.glob(os.path.join(train_dir, 'valid_*.pt')))
    vs = sorted(glob.glob(os.path.join(val_dir, 'shard_*.pt')))
    if (len(ts), len(tv)) != (expect['n_train_shards'], expect['n_valid_sidecars']) or len(vs) < expect.get('min_val_shards', 1):
        raise PreflightError(f"shard counts train={len(ts)} sidecars={len(tv)} val={len(vs)}; expected {expect['n_train_shards']}/{expect['n_valid_sidecars']}/>={expect.get('min_val_shards',1)}")
    tr, va = audit_shards(ts, seq_len, vocab), audit_shards(vs, seq_len, vocab)
    check_sidecars(train_dir, tr, seq_len)
    out = {'train_rows': sum(r['rows'] for r in tr), 'val_rows': sum(r['rows'] for r in va), 'records': tr + va}
    for k, got in (('train_rows', out['train_rows']), ('val_rows', out['val_rows'])):
        if k in expect and expect[k] != got: raise PreflightError(f"{k}={got}, expected {expect[k]}")
    out['tokenizer_sha256'] = sha256_file(tokenizer_path)
    if 'tokenizer_sha256' in expect and expect['tokenizer_sha256'] != out['tokenizer_sha256']:
        raise PreflightError(f"tokenizer sha mismatch actual={out['tokenizer_sha256']}")
    base = expect.get('baseline_records')
    if base is not None:
        key = lambda rs: [(r['name'], r['rows'], r['sha256']) for r in rs]   # positional: train and val shards can share basenames
        cur_k, base_k = key(out['records']), key(base)
        bad = [a[0] for a, b in zip(cur_k, base_k) if a != b]
        if bad or len(cur_k) != len(base_k): raise PreflightError(f"shard hashes differ from baseline: {bad[:5]} (n={len(cur_k)} vs {len(base_k)})")
    out['fingerprint'] = manifest_fingerprint(sorted(out['records'], key=lambda r: r['name']), tokenizer_path)
    return out

def record_baseline(layout):
    """Hashes to store (in a private location) after a verified mount, for later comparison. No bearer URLs."""
    return {'records': layout['records'], 'tokenizer_sha256': layout['tokenizer_sha256'], 'fingerprint': layout['fingerprint']}

# ---------- checkpoint ----------
REQUIRED = ('model_state_dict', 'optimizer_state_dict', 'resume_state', 'step')
def _opt_counters(osd):
    s = set()
    for st in osd['state'].values():
        if 'step' in st: s.add(int(st['step'].item() if hasattr(st['step'], 'item') else st['step']))
    return s

def unique_param_numel(sd):
    seen, n = set(), 0
    for v in sd.values():
        key = (v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape), tuple(v.stride()))
        if key in seen: continue
        seen.add(key); n += v.numel()
    return n

def inspect_checkpoint(path, schema, load=None):
    """schema: dict. Required: seqs_per_step. Optional: expect_step, expect_cursor, expect_unique_params, total_train_rows,
    fingerprint, need_scaler, allow_partial_batch. Equalities asserted are only those train.py's own resume validation asserts
    (resume_state.optimizer_step_count == optimizer counters, fingerprint, seqs_per_step, order_version); others are bounds."""
    import torch
    c = (load or (lambda p: torch.load(p, map_location='cpu', weights_only=True)))(path)
    missing = [k for k in REQUIRED if k not in c or c[k] is None] + (['scaler_state_dict'] if schema.get('need_scaler') and not c.get('scaler_state_dict') else [])
    if missing: raise PreflightError(f"checkpoint missing keys: {missing}; has {sorted(c.keys())}")
    rs = c['resume_state']
    if not isinstance(rs, dict): raise PreflightError('resume_state is not a dict (legacy checkpoint?)')
    for k in ('schema_version', 'order_version', 'seqs_per_step', 'sample_cursor', 'completed_updates', 'optimizer_step_count', 'fingerprint'):
        if k not in rs: raise PreflightError(f"resume_state lacks '{k}' (nested key path: checkpoint['resume_state'][...]); has {sorted(rs.keys())}")
    if rs['schema_version'] != 1 or rs['order_version'] != 'legacy-filtered-v1': raise PreflightError(f"unsupported resume schema {rs['schema_version']}/{rs['order_version']}")
    if rs['seqs_per_step'] != schema['seqs_per_step']: raise PreflightError(f"seqs_per_step {rs['seqs_per_step']} != {schema['seqs_per_step']}")
    counters = _opt_counters(c['optimizer_state_dict'])
    if len(counters) != 1 or rs['optimizer_step_count'] != next(iter(counters)): raise PreflightError(f"optimizer counters {sorted(counters)} vs resume_state {rs['optimizer_step_count']}")
    step, cur = c['step'], rs['sample_cursor']
    if cur < 0 or rs['completed_updates'] < 0: raise PreflightError('negative counters')
    bound = step * rs['seqs_per_step']
    if cur > bound or (cur != bound and not schema.get('allow_partial_batch')): raise PreflightError(f"cursor {cur} vs step*seqs_per_step {bound}")
    if rs['completed_updates'] > step: raise PreflightError('completed_updates > step')
    if 'total_train_rows' in schema and cur > schema['total_train_rows']: raise PreflightError('cursor beyond dataset')
    for k, got in (('expect_step', step), ('expect_cursor', cur)):
        if schema.get(k) is not None and schema[k] != got: raise PreflightError(f"{k[7:]} {got} != {schema[k]}")
    if 'fingerprint' in schema and rs['fingerprint'] != schema['fingerprint']: raise PreflightError('checkpoint data/tokenizer fingerprint != recomputed fingerprint')
    sd = c['model_state_dict']; uniq, naive = unique_param_numel(sd), sum(v.numel() for v in sd.values())
    if schema.get('expect_unique_params') is not None and uniq != schema['expect_unique_params']: raise PreflightError(f"unique state_dict numel {uniq} != {schema['expect_unique_params']} (naive sum {naive}; tied weights share storage)")
    bad = [k for k, v in sd.items() if v.is_floating_point() and not bool(v.float().isfinite().all())]
    if bad: raise PreflightError(f"non-finite weights in {bad[:3]}")
    return {'step': step, 'sample_cursor': cur, 'completed_updates': rs['completed_updates'], 'skipped_or_partial_info': step - rs['completed_updates'],
            'optimizer_step_count': rs['optimizer_step_count'], 'unique_numel': uniq, 'naive_numel': naive, 'val_loss': c.get('val_loss'), 'keys': sorted(c.keys())}

# ---------- memory: arithmetic components only; NOT a fit gate until calibrated against measured peaks ----------
def memory_components(params, micro_batch, seq, vocab):
    G = 1 << 30
    return {'weights_fp32_gb': params * 4 / G, 'grads_fp32_gb': params * 4 / G, 'adam_states_gb': params * 8 / G,
            'logits_fp16_gb': micro_batch * seq * vocab * 2 / G, 'logits_fp32_copy_gb': micro_batch * seq * vocab * 4 / G,
            'calibrated': False, 'note': 'components only; activations/allocator/framework overhead not modeled; no fit verdict'}

# ---------- credentials (defense in depth only) ----------
_PATS = [('hf_token', r'hf_[A-Za-z0-9]{30,}'), ('sk_style', r'sk-[A-Za-z0-9_-]{20,}'), ('google_key', r'AIza[0-9A-Za-z_-]{30,}'),
         ('private_key', r'-----BEGIN [A-Z ]*PRIVATE KEY-----'), ('kgat', r'KGAT_[0-9a-f]{20,}'), ('aws', r'AKIA[0-9A-Z]{16}'),
         ('literal_assign', r'''(?i)\b(?:hf_token|api_key|secret|password|token)\s*=\s*["'][^"'\s]{12,}["']''')]
def scan_secrets(text):
    """[(kind, line)] only; never returns matched text. Absence of findings is NOT proof of no secrets."""
    return [(k, i) for i, line in enumerate(text.splitlines(), 1) for k, p in _PATS if re.search(p, line)]

def check_cell_source(src, require_substrings=()):
    try: ast.parse(src)
    except SyntaxError as e: raise PreflightError(f"cell does not compile: line {e.lineno}")
    s = scan_secrets(src)
    if s: raise PreflightError(f"credential-like content (kind,line): {s}")
    miss = [r for r in require_substrings if r not in src]
    if miss: raise PreflightError(f"missing required guard(s): {miss}")
    return {'sha256': hashlib.sha256(src.encode()).hexdigest(), 'bytes': len(src.encode())}
