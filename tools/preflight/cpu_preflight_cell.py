"""CPU-only preflight driver (no GPU, no network, no secrets; never executes model output). Read-only on inputs.
Mandatory before any load: exact checkpoint sha256 + byte size, exact train.py sha256, tokenizer sha256, shard/sidecar counts.
torch.load is always weights_only=True and only runs AFTER the checkpoint hash and size match; no unsafe fallback exists."""
import json, os, time, glob
from cf_preflight import *

REQUIRED_KEYS = ('ckpt_sha256', 'ckpt_bytes', 'train_py_sha256', 'tokenizer_sha256', 'n_train_shards', 'n_valid_sidecars', 'seqs_per_step')
DEFAULT = dict(out='/kaggle/working/preflight_report.json', min_val_shards=1, train_rows=118125, val_rows=221,
               expect_step=None, expect_cursor=None, expect_unique_params=None, allow_partial_batch=False,
               baseline_records=None, micro_batch=8, seq_len=2048, vocab=32000, params=245924864)

def run(cfg):
    c = dict(DEFAULT, **cfg)
    missing = [k for k in REQUIRED_KEYS if c.get(k) is None]
    if missing: raise PreflightError(f"mandatory expectations not provided: {missing}")
    r = {'started': time.time(), 'checks': {}, 'scope': {}}; box = {}
    def step(name, fn):
        try: r['checks'][name] = {'ok': True, **(fn() or {})}
        except PreflightError as e: r['checks'][name] = {'ok': False, 'error': str(e)}
        except Exception as e: r['checks'][name] = {'ok': False, 'error': f"{type(e).__name__}: {str(e)[:300]}"}
    G = c['globs']
    for key in ('ckpt', 'train_py', 'tokenizer', 'train', 'val'):
        step('mount_' + key, lambda key=key: box.__setitem__(key, find_exactly_one(G[key], key)) or {'path': box[key]})
    if 'train_py' in box: step('source_sha', lambda: check_sha(box['train_py'], c['train_py_sha256'], 'train.py'))
    if all(k in box for k in ('train', 'val', 'tokenizer')):
        tokd = {k: v for k, v in c.items()}
        def data():
            L = check_data_layout(box['train'], box['val'], box['tokenizer'], dict(c, tokenizer_sha256=c['tokenizer_sha256']), vocab=c['vocab'], seq_len=c['seq_len'])
            box['fp'] = L['fingerprint']; box['layout'] = L; r['scope']['data'] = 'FULL: every train+val shard loaded, hashed, id-range checked; every sidecar checked'
            return {k: L[k] for k in ('train_rows', 'val_rows', 'tokenizer_sha256', 'fingerprint')}
        step('data_layout', data)
    if 'ckpt' in box:
        def sz():
            n = os.path.getsize(box['ckpt'])
            if n != c['ckpt_bytes']: raise PreflightError(f"checkpoint bytes {n} != {c['ckpt_bytes']}")
            return {'bytes': n}
        step('ckpt_size', sz)
        step('ckpt_sha', lambda: check_sha(box['ckpt'], c['ckpt_sha256'], 'checkpoint'))
        if r['checks'].get('ckpt_size', {}).get('ok') and r['checks'].get('ckpt_sha', {}).get('ok'):
            sch = {k: c[k] for k in ('seqs_per_step', 'expect_step', 'expect_cursor', 'expect_unique_params', 'allow_partial_batch') if c.get(k) is not None}
            sch['total_train_rows'] = c['train_rows']
            if 'fp' in box: sch['fingerprint'] = box['fp']
            r['scope']['checkpoint'] = 'FULL: all state_dict tensors finite-checked; optimizer counters read; weights_only=True'
            step('checkpoint', lambda: inspect_checkpoint(box['ckpt'], sch))
        else: r['checks']['checkpoint'] = {'ok': False, 'error': 'not loaded: size/sha precondition failed'}
    r['memory_components'] = memory_components(c['params'], c['micro_batch'], c['seq_len'], c['vocab'])
    r['all_ok'] = all(v['ok'] for v in r['checks'].values()) and 'checkpoint' in r['checks']
    if 'layout' in box: r['baseline_to_store_privately'] = record_baseline(box['layout'])
    os.makedirs(os.path.dirname(c['out']), exist_ok=True); json.dump(r, open(c['out'], 'w'), indent=1, default=str)
    return r
