# A3-v2 CPU data preparation. Requires published A3 scripts/config on master.
# Evol/Glaive starts are intentionally reused; code offsets are approximate.
# Preserves the Stage A tokenizer and refuses an existing stageA3-v2 folder.
import os, sys, subprocess, pathlib, shutil, glob, json, time
os.environ.update(HF_HUB_DISABLE_PROGRESS_BARS='1', TOKENIZERS_PARALLELISM='false', CF_SKIP_STAGE_A_DOCS='0')
from kaggle_secrets import UserSecretsClient
TOKEN = UserSecretsClient().get_secret('HF_TOKEN'); os.environ['HF_TOKEN'] = TOKEN
REPO = 'Yash1bajpai/CodeForge-250M-rotation'
CF = pathlib.Path('/kaggle/working/CodeForge-250M')
def run(cmd, **kw):
    print('$', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, **kw)
    if r.returncode: raise SystemExit(f'FAILED rc={r.returncode}: {cmd[:3]}')
run(['pip', '-q', 'install', 'datasets==2.21.0', 'huggingface_hub', 'pyyaml', 'requests'])
run(['git', 'clone', '-q', 'https://github.com/Yash1bajpai/CodeForge-250M.git', str(CF)])
run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=CF)
sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=CF, capture_output=True, text=True).stdout.strip()
print('code commit', sha, flush=True)
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=TOKEN)
assert not any(p.startswith('stageA3-v2/') for p in api.list_repo_files(REPO)), 'stageA3 already exists; refusing overwrite' 
tdir = snapshot_download(REPO, token=TOKEN, allow_patterns=['data/tokenizer/*'])
(CF / 'data').mkdir(exist_ok=True)
shutil.copytree(pathlib.Path(tdir) / 'data/tokenizer', CF / 'data/tokenizer', dirs_exist_ok=True)
names = [a['content'] for a in json.load(open(CF / 'data/tokenizer/tokenizer.json')).get('added_tokens', [])]
need = ['<|endoftext|>', '<|unk|>', '<|pad|>', '<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>', '<|tool_call|>', '<|tool_result|>', '<|thinking|>', '<|json_start|>', '<|json_end|>']
assert all(t in names for t in need), 'Stage A tokenizer missing specials'
run([sys.executable, '-c', "from data.download_stack import download_curated_stack as d; d('configs/config_250M_a3.yaml')"], cwd=CF, env={**os.environ, 'PYTHONPATH': str(CF)})
stats=json.load(open(CF/'data/raw/source_stats.json'))
print('SOURCE_STATS',json.dumps(stats),flush=True)
assert stats['evol-codealpaca']['documents'] > 0
assert stats['glaive-function-calling']['written_chars'] >= 20_000_000
run([sys.executable, 'data/filter_quality.py'], cwd=CF); shutil.rmtree(CF / 'data/raw', ignore_errors=True)
run([sys.executable, 'data/deduplicate.py'], cwd=CF); shutil.rmtree(CF / 'data/filtered', ignore_errors=True)
retained={}
for f in (CF/'data/dedup').glob('*_dedup.jsonl'):
    retained[f.name.split('_')[0]]=sum(1 for _ in open(f))
print('RETAINED_DOCS',json.dumps(retained),flush=True)
assert retained.get('evol-codealpaca',0)>0, 'No instruction docs retained'
run([sys.executable, 'data/tokenize_dataset.py'], cwd=CF)
import torch
shards = sorted(glob.glob(str(CF / 'data/tokenized/shard_*.pt')))
fim_id = None
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast.from_pretrained(str(CF / 'data/tokenizer'))
fim_id = tok.convert_tokens_to_ids('<|fim_prefix|>')
rows = fim_rows = 0
for p in shards:
    t = torch.load(p); rows += t.shape[0]; fim_rows += int((t == fim_id).any(dim=1).sum())
print(f'shards={len(shards)} rows={rows} rows_with_fim_prefix={fim_rows} ({fim_rows/max(rows,1):.1%})', flush=True)
assert len(shards) >= 20, 'too little data'
assert rows >= 180 * 256 + 1000, 'Not enough single-pass rows for configured 180 steps'
manifest = {'code_commit': sha, 'shards': len(shards), 'rows': rows, 'rows_with_fim_prefix': fim_rows, 'time': time.time(), 'source_stats': stats, 'retained_docs': retained, 'instruction_replay': 'Evol and Glaive offsets zero intentionally; not fresh-only'}
json.dump(manifest, open('/kaggle/working/stageA3_v2_manifest.json', 'w'))
api = HfApi(token=TOKEN)
api.upload_folder(repo_id=REPO, folder_path=str(CF / 'data/tokenized'), path_in_repo='stageA3-v2/data/tokenized', commit_message='Stage A3-v2 FIM-fixed shards (new folder, nothing overwritten)')
api.upload_file(path_or_fileobj='/kaggle/working/stageA3_v2_manifest.json', path_in_repo='stageA3-v2/manifest.json', repo_id=REPO, commit_message='Stage A3-v2 manifest')
print('DONE data prep', manifest, flush=True)
