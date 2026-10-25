import os, tempfile, torch, shutil
from cf_preflight import *
import cpu_preflight_cell as cell
def raises(f, *a, **k):
    try: f(*a, **k)
    except PreflightError as e: return str(e)
    raise AssertionError('expected PreflightError')
d = tempfile.mkdtemp(); root = f'{d}/input/ds'
tr, va = f'{root}/train_ready', f'{root}/val_ready'; tokd = f'{root}/hf/data/tokenizer'; ckd = f'{root}/proj/checkpoints/gate'
for p in (tr, va, tokd, ckd): os.makedirs(p)
open(f'{tokd}/tokenizer.json', 'w').write('{"tok":1}')
rows = [4, 4, 3]
for i, n in enumerate(rows):
    torch.save(torch.randint(0, 100, (n, 16), dtype=torch.int32).to(torch.uint16), f'{tr}/shard_{i:04d}.pt'); torch.save({'valid_len': torch.full((n,), 16, dtype=torch.int32)}, f'{tr}/valid_{i:04d}.pt')
torch.save(torch.randint(0, 100, (2, 16), dtype=torch.int32).to(torch.uint16), f'{va}/shard_0000.pt')
tokp = f'{tokd}/tokenizer.json'
E_ = dict(n_train_shards=3, n_valid_sidecars=3, min_val_shards=1, train_rows=11, val_rows=2)
L = check_data_layout(tr, va, tokp, E_, vocab=200, seq_len=16); assert L['train_rows'] == 11
# negatives
assert 'counts' in raises(check_data_layout, tr, va, tokp, dict(E_, n_train_shards=4), vocab=200, seq_len=16)
assert 'train_rows' in raises(check_data_layout, tr, va, tokp, dict(E_, train_rows=12), vocab=200, seq_len=16)
assert 'out-of-vocab' in raises(check_data_layout, tr, va, tokp, E_, vocab=50, seq_len=16)
assert 'bad shard' in raises(check_data_layout, tr, va, tokp, E_, vocab=200, seq_len=32)
assert 'tokenizer sha' in raises(check_data_layout, tr, va, tokp, dict(E_, tokenizer_sha256='0' * 64), vocab=200, seq_len=16)
vp = f'{tr}/valid_0001.pt'; good = torch.load(vp, weights_only=True)
torch.save({'valid_len': torch.tensor([1, 16, 16, 16], dtype=torch.int32)}, vp); assert 'bad valid_len' in raises(check_data_layout, tr, va, tokp, E_, vocab=200, seq_len=16)
torch.save({'valid_len': torch.full((3,), 16, dtype=torch.int32)}, vp); assert 'bad valid_len' in raises(check_data_layout, tr, va, tokp, E_, vocab=200, seq_len=16)
torch.save({'x': 1}, vp); assert "lacks 'valid_len'" in raises(check_data_layout, tr, va, tokp, E_, vocab=200, seq_len=16)
os.remove(vp); torch.save(good, f'{tr}/valid_0009.pt'); assert 'missing sidecar' in raises(check_data_layout, tr, va, tokp, E_, vocab=200, seq_len=16)
os.remove(f'{tr}/valid_0009.pt'); torch.save(good, vp)
base = record_baseline(L); assert check_data_layout(tr, va, tokp, dict(E_, baseline_records=base['records']), vocab=200, seq_len=16)['fingerprint'] == L['fingerprint']
tampered = [dict(r) for r in base['records']]; tampered[0]['sha256'] = '0' * 64
assert 'differ from baseline' in raises(check_data_layout, tr, va, tokp, dict(E_, baseline_records=tampered), vocab=200, seq_len=16)
# fingerprint algorithm: order-independent of input list since sorted by name
assert manifest_fingerprint(sorted(L['records'], key=lambda r: r['name']), tokp) == L['fingerprint']
# checkpoint with tied weights + non-persistent buffer
class M(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.emb = torch.nn.Embedding(10, 4); s.head = torch.nn.Linear(4, 10, bias=False); s.head.weight = s.emb.weight; s.register_buffer('inv', torch.ones(2), persistent=False)
m = M(); opt = torch.optim.AdamW(m.parameters()); m.head(m.emb(torch.tensor([1]))).sum().backward(); opt.step()
assert sum(v.numel() for v in m.state_dict().values()) == 80 and unique_param_numel(m.state_dict()) == 40  # tied counted once; buffer absent
def mk(step=100, cursor=25600, comp=100, **rs):
    r = {'schema_version': 1, 'order_version': 'legacy-filtered-v1', 'seqs_per_step': 256, 'sample_cursor': cursor, 'completed_updates': comp, 'optimizer_step_count': 1, 'fingerprint': L['fingerprint']}; r.update(rs)
    return {'model_state_dict': m.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'step': step, 'resume_state': r, 'val_loss': 1.7, 'scaler_state_dict': {}}
n = [0]
def save(c, name=None):
    n[0] += 1; q = f'{d}/c{n[0]}.pt'; torch.save(c, q); return q
S = dict(seqs_per_step=256, expect_step=100, expect_cursor=25600, expect_unique_params=40, total_train_rows=118125, fingerprint=L['fingerprint'])
i = inspect_checkpoint(save(mk()), S); assert i['unique_numel'] == 40 and i['naive_numel'] == 80
# real-training semantics: scaler-skipped updates allowed (completed_updates < step); partial final batch needs explicit allow
assert inspect_checkpoint(save(mk(comp=97)), S)['skipped_or_partial_info'] == 3
assert 'cursor' in raises(inspect_checkpoint, save(mk(cursor=25500)), dict(S, expect_cursor=None))
assert inspect_checkpoint(save(mk(cursor=25500)), dict(S, expect_cursor=None, allow_partial_batch=True))
assert 'completed_updates' in raises(inspect_checkpoint, save(mk(comp=101)), S)
assert 'cursor' in raises(inspect_checkpoint, save(mk(cursor=25856)), S)           # beyond step*seqs
assert 'nested key path' in raises(inspect_checkpoint, save({**mk(), 'resume_state': {'sample_cursor': 1}}), S)
assert 'resume_state' in raises(inspect_checkpoint, save({k: v for k, v in mk().items() if k != 'resume_state'}), S)
assert 'seqs_per_step' in raises(inspect_checkpoint, save(mk()), dict(S, seqs_per_step=128))
assert 'unsupported' in raises(inspect_checkpoint, save(mk(order_version='x')), S)
assert 'optimizer counters' in raises(inspect_checkpoint, save(mk(optimizer_step_count=9)), S)
assert 'fingerprint' in raises(inspect_checkpoint, save(mk(fingerprint='bad')), S)
assert 'tied' in raises(inspect_checkpoint, save(mk()), dict(S, expect_unique_params=80))
assert 'step 101' in raises(inspect_checkpoint, save(mk(step=101, cursor=25856)), S)
bad = mk(); bad['model_state_dict'] = {k: v.clone() for k, v in m.state_dict().items()}; next(iter(bad['model_state_dict'].values()))[0, 0] = float('nan')
assert 'non-finite' in raises(inspect_checkpoint, save(bad), dict(S, expect_unique_params=None))
assert 'scaler' in raises(inspect_checkpoint, save(mk()), dict(S, need_scaler=True))
# memory components are arithmetic only
mc = memory_components(245924864, 8, 2048, 32000); assert mc['calibrated'] is False and 'fits' not in mc and abs(mc['logits_fp16_gb'] - 8*2048*32000*2/2**30) < 1e-9
# secrets
fake = 'hf_' + 'a' * 34
r = scan_secrets(f'HF_TOKEN = "{fake}"\n'); assert r and fake not in str(r) and scan_secrets('os.environ.get("HF_TOKEN")\n') == []
msg = raises(check_cell_source, f'a="{fake}"\n'); assert fake not in msg
assert 'compile' in raises(check_cell_source, 'def (:') and 'missing required' in raises(check_cell_source, 'x=1\n', ('guard',))
# end-to-end driver on synthetic mounts
import cpu_preflight_cell as cell
ck = mk(step=3, cursor=6, comp=3, seqs_per_step=2); cp = f'{ckd}/latest_checkpoint.pt'; torch.save(ck, cp)
os.makedirs(f'{root}/proj/training'); open(f'{root}/proj/training/train.py', 'w').write('print(1)\n')
cfg = dict(out=f'{d}/out/report.json', globs=dict(ckpt=f'{d}/input/ds/proj/checkpoints/gate/latest_checkpoint.pt', train_py=f'{d}/input/ds/proj/training/train.py',
           tokenizer=f'{d}/input/ds/hf/data/tokenizer/tokenizer.json', train=f'{d}/input/ds/train_ready', val=f'{d}/input/ds/val_ready'),
           ckpt_sha256=sha256_file(cp), ckpt_bytes=os.path.getsize(cp), train_py_sha256=sha256_file(f'{root}/proj/training/train.py'),
           tokenizer_sha256=sha256_file(tokp), n_train_shards=3, n_valid_sidecars=3, train_rows=11, val_rows=2, seqs_per_step=2,
           expect_step=3, expect_cursor=6, expect_unique_params=40, vocab=200, seq_len=16, params=40)
rep = cell.run(cfg); assert rep['all_ok'], rep['checks']; assert os.path.exists(cfg['out']) and 'FULL' in rep['scope']['data']
# mandatory expectations
for k in cell.REQUIRED_KEYS:
    assert 'mandatory' in raises(cell.run, {kk: v for kk, v in cfg.items() if kk != k})
# no load before hash/size match: patch torch.load to count calls on the checkpoint path
calls = []; orig = torch.load
def spy(p, *a, **k): calls.append(str(p)); return orig(p, *a, **k)
torch.load = spy
rep = cell.run(dict(cfg, ckpt_sha256='0' * 64)); assert not rep['all_ok'] and 'mismatch' in rep['checks']['ckpt_sha']['error'] and rep['checks']['checkpoint']['error'].startswith('not loaded')
assert not [p for p in calls if p.endswith('latest_checkpoint.pt')], calls
rep = cell.run(dict(cfg, ckpt_bytes=1)); assert not rep['all_ok'] and not [p for p in calls if p.endswith('latest_checkpoint.pt')]
rep = cell.run(cfg); assert rep['all_ok'] and [p for p in calls if p.endswith('latest_checkpoint.pt')]
torch.load = orig
assert not cell.run(dict(cfg, train_py_sha256='0' * 64))['all_ok']
shutil.copytree(root, f'{d}/input/dup'); bad = dict(cfg, globs=dict(cfg['globs'], ckpt=f'{d}/input/**/latest_checkpoint.pt'))
rep = cell.run(bad); assert not rep['all_ok'] and 'found 2' in rep['checks']['mount_ckpt']['error']
print('all preflight fixtures pass (fixture-only; not validated on real shards/checkpoints)')
