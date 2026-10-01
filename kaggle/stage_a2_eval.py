# Stage A2 eval (Kaggle GPU notebook, secret HF_TOKEN). Same protocol as the Stage A report
# (greedy, CF_MB_STYLE=named, defaults). Also a FIM-leak probe with special-token masking OFF
# (first 60 problems) for Stage A vs Stage A2. Results upload under eval/ in the HF repo.
import os, json, runpy, shutil, re
os.environ.update(HF_HUB_DISABLE_PROGRESS_BARS='1', TOKENIZERS_PARALLELISM='false', PYTORCH_ALLOC_CONF='expandable_segments:True')
from kaggle_secrets import UserSecretsClient
TOKEN = UserSecretsClient().get_secret('HF_TOKEN'); os.environ['HF_TOKEN'] = TOKEN
from huggingface_hub import HfApi
REPO = 'Yash1bajpai/CodeForge-250M-rotation'
A2_REV = HfApi(token=TOKEN).model_info(REPO).sha
print('A2 rev', A2_REV, flush=True)
EVAL = '/kaggle/working/_src/evaluation/codeforge_eval.py'
OUT = '/kaggle/working'
def run(env):
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        runpy.run_path(EVAL, run_name='__main__')
    finally:
        for k, v in keep.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
def leak(tag):
    txt = open(f'{OUT}/humaneval_samples.json').read()
    n = len(re.findall(r'<\|(fim_prefix|fim_middle|fim_suffix|tool_call|tool_result|thinking|json_start|json_end)\|>', txt))
    print(f'LEAK_PROBE {tag}: special-token occurrences in 60 HumanEval completions = {n}', flush=True)
    return n
res = {}
os.environ['CF_MB_STYLE'] = 'named'
# probe: Stage A (default rev), masking off
run({'CF_LIMIT': '60', 'CF_BAN': '0', 'CF_NOUPLOAD': '1', 'CF_DIAG': '0'})
res['stage_a_leak'] = leak('stage_a')
# probe: Stage A2, masking off
run({'CF_LIMIT': '60', 'CF_BAN': '0', 'CF_NOUPLOAD': '1', 'CF_DIAG': '0', 'CF_REV': A2_REV, 'CF_CKPT_FILE': 'stageA2/latest_checkpoint.pt'})
res['stage_a2_leak'] = leak('stage_a2')
# full eval, Stage A2, same protocol as Stage A report; uploads to eval/stage_a2
run({'CF_REV': A2_REV, 'CF_CKPT_FILE': 'stageA2/latest_checkpoint.pt', 'CF_EVAL_TAG': 'stage_a2'})
res['stage_a2_eval'] = json.load(open(f'{OUT}/eval_results.json'))
print('SUMMARY', json.dumps(res), flush=True)
HfApi(token=TOKEN).upload_file(path_or_fileobj=json.dumps(res, indent=1).encode(), path_in_repo='eval/stage_a2/summary.json', repo_id=REPO, commit_message='Stage A2 eval summary')
