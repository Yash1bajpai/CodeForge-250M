"""Single-T4 A3 launcher. Start only on the owner-approved machine and budget."""
import os,sys,time,subprocess,pathlib,shutil,json
START=time.time()
# Provisional 1h trainer cap and 1.5h absolute deadline.
# Final launch requires a live quota check including setup and checkpoint uploads.
os.environ.update(CUDA_VISIBLE_DEVICES='0',CF_TELEMETRY_DISABLED='1',WANDB_MODE='disabled',
                  TOKENIZERS_PARALLELISM='false',CF_ABSOLUTE_STOP_EPOCH=str(START+1.5*3600))
import torch
assert torch.cuda.is_available(), 'No GPU: refusing training'
assert torch.cuda.get_device_name(0).find('T4')>=0,'Approved T4 required'
from kaggle_secrets import UserSecretsClient
os.environ['HF_TOKEN']=UserSecretsClient().get_secret('HF_TOKEN')
from huggingface_hub import HfApi,snapshot_download,hf_hub_download
REPO='Yash1bajpai/CodeForge-250M-rotation'
api=HfApi(token=os.environ['HF_TOKEN'])
rev=api.model_info(REPO).sha
root=pathlib.Path(__file__).resolve().parents[1]
files=api.list_repo_files(REPO,revision=rev)
assert not any(p.startswith('stageA3-v2/') and 'checkpoint' in p for p in files),'A3 checkpoint already exists'
manifest_path=hf_hub_download(REPO,'stageA3-v2/manifest.json',revision=rev,token=os.environ['HF_TOKEN'])
m=json.load(open(manifest_path))
assert m['rows']==60459 and m['shards']==61
assert m['retained_docs']['glaive-function-calling']==8609
assert len([p for p in files if p.startswith('stageA3-v2/data/tokenized/shard_')])==61
print('HF_READBACK',rev,m['rows'],m['shards'],flush=True)
d=snapshot_download(REPO,revision=rev,allow_patterns=['stageA3-v2/data/tokenized/*','data/tokenizer/*'],token=os.environ['HF_TOKEN'])
cp=hf_hub_download(REPO,'stageA2/latest_checkpoint.pt',revision=rev,token=os.environ['HF_TOKEN'])
c=torch.load(cp,map_location='cpu',weights_only=True)
assert c['step']==150,'Unexpected A2 source';del c
for name,src in [('tokenized','stageA3-v2/data/tokenized'),('tokenizer','data/tokenizer')]:
    dst=root/'data'/name
    if dst.is_symlink():dst.unlink()
    elif dst.exists():raise RuntimeError(f'Refusing to overwrite existing {dst}; stage manually')
    dst.symlink_to(pathlib.Path(d)/src,target_is_directory=True)
subprocess.run([sys.executable,'training/train.py','--config','configs/config_250M_a3.yaml',
                '--init_from',cp,'--max_hours','1.0'],cwd=root,check=True)
print('TRAINING_FINISHED: machine must now be stopped by supervisor',flush=True)
