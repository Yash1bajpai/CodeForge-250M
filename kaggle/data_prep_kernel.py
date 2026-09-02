#!/usr/bin/env python
# Kaggle CPU kernel (FREE, no GPU quota): runs the data pipeline and pushes
# tokenized shards + tokenizer to the private KAGGLE_DATA_DATASET.
# Resumable: raw/filtered/dedup outputs are zipped and re-downloaded between
# the 4-hour-restart CPU sessions until the target is met.
import os, sys, subprocess, shutil, glob, json

CF = "/kaggle/working/CodeForge-250M"
DATA_DS = os.environ.get("KAGGLE_DATA_DATASET", "yashbajpai2027/codeforge-data")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# HF token injected at push time via env; kernel reads it from KAGGLE secrets-free env
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
if not os.environ["HF_TOKEN"]:
    # fallback: token file bundled with the kernel push
    _tok_file = "/kaggle/working/hf_token.txt"
    if os.path.exists(_tok_file):
        os.environ["HF_TOKEN"] = open(_tok_file).read().strip()
        os.environ["HF_TOKEN"] = os.environ["HF_TOKEN"]

print("=== [Data-prep kernel] boot ===", flush=True)

def kaggle_pull(slug, dest):
    os.makedirs(dest, exist_ok=True)
    r = subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", dest, "--unzip", "-o"])
    return r.returncode == 0

# clone code (kernel push may also bundle it)
if not os.path.exists(f"{CF}/data/download_stack.py"):
    subprocess.run(["git", "clone", "-q", "https://github.com/Yash1bajpai/CodeForge-250M.git", CF], check=False)

# restore previous partial state
kaggle_pull(DATA_DS, "/tmp/data_in")
for sub in ["raw", "filtered", "dedup", "tokenizer", "tokenized"]:
    src, dst = f"/tmp/data_in/{sub}", f"{CF}/data/{sub}"
    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(f"{src}/*"):
            shutil.copy(f, dst)

# datasets 3.x dropped script-loading (commitpackft needs 2.x); pin BEFORE imports in pipeline
subprocess.run(["pip", "install", "-q", "datasets==2.21.0", "tokenizers", "transformers", "tiktoken"], check=False)

# stages 1-3: download -> filter -> dedup (all CPU, all streaming)
# DISK BUDGET: Kaggle gives ~20GB working. Raw is ~9GB, so we must delete each
# stage's inputs as soon as the next stage finishes consuming them.
import shutil as _sh
def _dirsize(p):
    return sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(p) for f in fs)

print(f"[disk] working set before download: {_dirsize('/kaggle/working')/1e9:.1f} GB", flush=True)
subprocess.run([sys.executable, f"{CF}/data/download_stack.py"], cwd=CF, check=False)
print(f"[disk] after download: {_dirsize('/kaggle/working')/1e9:.1f} GB", flush=True)

subprocess.run([sys.executable, f"{CF}/data/filter_quality.py"], cwd=CF, check=False)
_sh.rmtree(f"{CF}/data/raw", ignore_errors=True)          # raw consumed by filter
print(f"[disk] after filter (raw deleted): {_dirsize('/kaggle/working')/1e9:.1f} GB", flush=True)

subprocess.run([sys.executable, f"{CF}/data/deduplicate.py"], cwd=CF, check=False)
_sh.rmtree(f"{CF}/data/filtered", ignore_errors=True)     # filtered consumed by dedup
print(f"[disk] after dedup (filtered deleted): {_dirsize('/kaggle/working')/1e9:.1f} GB", flush=True)

# stage 4: tokenizer — train fresh UNLESS the restored one passes the 11-token check
REQUIRED_SPECIALS = ["<|endoftext|>", "<|unk|>", "<|pad|>", "<|fim_prefix|>", "<|fim_middle|>",
                     "<|fim_suffix|>", "<|tool_call|>", "<|tool_result|>", "<|thinking|>",
                     "<|json_start|>", "<|json_end|>"]

def tokenizer_is_valid(path):
    if not os.path.exists(path):
        return False
    try:
        import json as _json
        d = _json.load(open(path))
        names = [a["content"] for a in d.get("added_tokens", [])]
        return all(t in names for t in REQUIRED_SPECIALS)
    except Exception:
        return False

tok_path = f"{CF}/data/tokenizer/tokenizer.json"
if not tokenizer_is_valid(tok_path):
    if os.path.exists(tok_path):
        print("[Tokenizer] restored tokenizer FAILED 11-token check — retraining", flush=True)
    _sh.rmtree(f"{CF}/data/tokenizer", ignore_errors=True)
    subprocess.run([sys.executable, f"{CF}/data/train_tokenizer.py"], cwd=CF, check=False)
# HARD GATE: refuse to tokenize with a broken tokenizer
if not tokenizer_is_valid(tok_path):
    raise SystemExit("FATAL: tokenizer still missing special tokens after training. "
                     "Refusing to build shards (run #1 B3 bug).")
print("[Tokenizer] verified: all 11 special tokens present", flush=True)

# stage 5: tokenize to uint16 shards (fixed random seed for the 50% FIM split)
subprocess.run([sys.executable, f"{CF}/data/tokenize_dataset.py"], cwd=CF, check=False)
n_shards = len(glob.glob(f"{CF}/data/tokenized/shard_*.pt"))
# dedup consumed by tokenize: free its ~8GB before the upload
if n_shards > 0:
    _sh.rmtree(f"{CF}/data/dedup", ignore_errors=True)
print(f"[disk] after tokenize (dedup deleted): {_dirsize('/kaggle/working')/1e9:.1f} GB", flush=True)

# report + push: only what training needs (tokenized + tokenizer), directly from
# the working dir — no /tmp copies (tmpfs copy of GBs can OOM the session).
print(f"=== [Data-prep] {n_shards} shards ready ===", flush=True)

meta = f"""{{
  "title": "codeforge-data",
  "id": "{DATA_DS}",
  "licenses": [{{"name": "CC0-1.0"}}]
}}"""
with open(f"{CF}/data/dataset-metadata.json", "w") as f:
    f.write(meta)
# keep ONLY push dirs inside data/ — remove stray metadata json from glob paths above
r = subprocess.run(["kaggle", "datasets", "version", "-p", f"{CF}/data",
                    "-m", f"prep-state-{n_shards}-shards", "--dir-mode", "zip"])
print(f"=== [Data-prep] push rc={r.returncode} ===", flush=True)
