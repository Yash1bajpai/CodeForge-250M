# Stage A2 data prep. Kaggle CPU notebook (no GPU quota). Credentials: Kaggle secret HF_TOKEN.
# Builds a SMALL fresh dataset with the FIM fix (FIM only on code corpora), skipping the
# documents Stage A already saw, using the SAME tokenizer as Stage A. Uploads shards to a NEW
# folder in the private HF repo (stageA2/data/tokenized). Nothing existing is overwritten.
import os, sys, subprocess, pathlib, shutil, glob, json, time
os.environ.update(HF_HUB_DISABLE_PROGRESS_BARS='1', TOKENIZERS_PARALLELISM='false', CF_SKIP_STAGE_A_DOCS='1')
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
sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=CF, capture_output=True, text=True).stdout.strip()
print('code commit', sha, flush=True)
from huggingface_hub import HfApi, snapshot_download
tdir = snapshot_download(REPO, token=TOKEN, allow_patterns=['data/tokenizer/*'])
(CF / 'data').mkdir(exist_ok=True)
shutil.copytree(pathlib.Path(tdir) / 'data/tokenizer', CF / 'data/tokenizer', dirs_exist_ok=True)
names = [a['content'] for a in json.load(open(CF / 'data/tokenizer/tokenizer.json')).get('added_tokens', [])]
need = ['<|endoftext|>', '<|unk|>', '<|pad|>', '<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>', '<|tool_call|>', '<|tool_result|>', '<|thinking|>', '<|json_start|>', '<|json_end|>']
assert all(t in names for t in need), 'Stage A tokenizer missing specials'
run([sys.executable, '-c', "from data.download_stack import download_curated_stack as d; d('configs/config_250M_a2.yaml')"], cwd=CF, env={**os.environ, 'PYTHONPATH': str(CF)})
run([sys.executable, 'data/filter_quality.py'], cwd=CF); shutil.rmtree(CF / 'data/raw', ignore_errors=True)
run([sys.executable, 'data/deduplicate.py'], cwd=CF); shutil.rmtree(CF / 'data/filtered', ignore_errors=True)
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
manifest = {'code_commit': sha, 'shards': len(shards), 'rows': rows, 'rows_with_fim_prefix': fim_rows, 'time': time.time()}
json.dump(manifest, open('/kaggle/working/stageA2_manifest.json', 'w'))
api = HfApi(token=TOKEN)
api.upload_folder(repo_id=REPO, folder_path=str(CF / 'data/tokenized'), path_in_repo='stageA2/data/tokenized', commit_message='Stage A2 FIM-fixed shards (new folder, nothing overwritten)')
api.upload_file(path_or_fileobj='/kaggle/working/stageA2_manifest.json', path_in_repo='stageA2/manifest.json', repo_id=REPO, commit_message='Stage A2 manifest')
print('DONE data prep', manifest, flush=True)
