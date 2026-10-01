# Stage A2 training. Kaggle GPU notebook (single T4, one account). Secret: HF_TOKEN.
# Warm start (weights only) from the Stage A checkpoint, fresh optimizer, low LR, FIM-fixed data.
# The Stage A checkpoint and everything else in the HF repo is read-only here; outputs go to stageA2/.
import os, sys, subprocess, pathlib, shutil, glob, json, time
START = time.time()
os.environ.update(HF_HUB_DISABLE_PROGRESS_BARS='1', WANDB_MODE='disabled', CF_TELEMETRY_DISABLED='1',
                  CUDA_VISIBLE_DEVICES='0', TOKENIZERS_PARALLELISM='false',
                  PYTORCH_ALLOC_CONF='expandable_segments:True')
os.environ['CF_ABSOLUTE_STOP_EPOCH'] = str(START + 6.5 * 3600)
from kaggle_secrets import UserSecretsClient
TOKEN = UserSecretsClient().get_secret('HF_TOKEN'); os.environ['HF_TOKEN'] = TOKEN
REPO = 'Yash1bajpai/CodeForge-250M-rotation'
CF = pathlib.Path('/kaggle/working/CodeForge-250M')
def run(cmd, **kw):
    print('$', ' '.join(cmd), flush=True); return subprocess.run(cmd, **kw)
import torch
assert torch.cuda.is_available(), 'No CUDA GPU; refusing CPU training'
run(['pip', '-q', 'install', 'huggingface_hub', 'pyyaml'])
if run(['git', 'clone', '-q', 'https://github.com/Yash1bajpai/CodeForge-250M.git', str(CF)]).returncode: raise SystemExit('clone failed')
sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=CF, capture_output=True, text=True).stdout.strip(); print('code', sha, flush=True)
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
api = HfApi(token=TOKEN)
head = api.model_info(REPO).sha
D = snapshot_download(REPO, revision=head, token=TOKEN, allow_patterns=['stageA2/data/tokenized/*', 'data/tokenizer/*'])
CP = hf_hub_download(REPO, 'latest_checkpoint.pt', revision=head, token=TOKEN)
_c = torch.load(CP, map_location='cpu', weights_only=True); print('Stage A ckpt step', _c.get('step'), 'val_loss', _c.get('val_loss'), flush=True)
assert _c.get('step') == 2898, 'unexpected source checkpoint step'; del _c
for sub, src in [('tokenized', pathlib.Path(D) / 'stageA2/data/tokenized'), ('tokenizer', pathlib.Path(D) / 'data/tokenizer')]:
    dst = CF / 'data' / sub
    if dst.exists(): shutil.rmtree(dst) if not dst.is_symlink() else dst.unlink()
    dst.symlink_to(src, target_is_directory=True)
print('shards', len(glob.glob(str(CF / 'data/tokenized/shard_*.pt'))), flush=True)
OUT = CF / 'checkpoints/CodeForge-250M-a2'
rc = run([sys.executable, 'training/train.py', '--config', 'configs/config_250M_a2.yaml', '--init_from', CP, '--max_hours', '5.5'], cwd=CF).returncode
print('train rc', rc, flush=True)
try:
    for f in ['latest_checkpoint.pt']:
        if (OUT / f).exists():
            api.upload_file(path_or_fileobj=str(OUT / f), path_in_repo='stageA2/latest_checkpoint.pt', repo_id=REPO, commit_message='Stage A2 checkpoint (new path; Stage A checkpoint untouched)')
    for lg in [CF / 'training.log', CF / 'metrics.json']:
        if lg.exists(): api.upload_file(path_or_fileobj=str(lg), path_in_repo=f'stageA2/{lg.name}', repo_id=REPO, commit_message='Stage A2 log')
    print('uploaded stageA2 outputs', flush=True)
except Exception as e:
    print('UPLOAD FAILED', e, flush=True)
    raise
sys.exit(rc)
