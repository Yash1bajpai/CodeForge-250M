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

# stage 4: tokenizer only if not yet trained (B3: all 11 special tokens)
if not os.path.exists(f"{CF}/data/tokenizer/tokenizer.json"):
    subprocess.run([sys.executable, f"{CF}/data/train_tokenizer.py"], cwd=CF, check=False)

# stage 5: tokenize to uint16 shards (fixed random seed for the 50% FIM split)
subprocess.run([sys.executable, f"{CF}/data/tokenize_dataset.py"], cwd=CF, check=False)

# report + push everything
n_shards = len(glob.glob(f"{CF}/data/tokenized/shard_*.pt"))
print(f"=== [Data-prep] {n_shards} shards ready ===", flush=True)

meta = f"""{{
  "title": "codeforge-data",
  "id": "{DATA_DS}",
  "licenses": [{{"name": "CC0-1.0"}}]
}}"""
os.makedirs("/tmp/data_out", exist_ok=True)
with open("/tmp/data_out/dataset-metadata.json", "w") as f:
    f.write(meta)
for sub in ["raw", "filtered", "dedup", "tokenizer", "tokenized"]:
    src = f"{CF}/data/{sub}"
    if os.path.isdir(src) and glob.glob(f"{src}/*"):
        os.makedirs(f"/tmp/data_out/{sub}", exist_ok=True)
        for f in glob.glob(f"{src}/*"):
            shutil.copy(f, f"/tmp/data_out/{sub}/")
r = subprocess.run(["kaggle", "datasets", "version", "-p", "/tmp/data_out",
                    "-m", f"prep-state-{n_shards}-shards", "--dir-mode", "zip"])
print(f"=== [Data-prep] push rc={r.returncode} ===", flush=True)
