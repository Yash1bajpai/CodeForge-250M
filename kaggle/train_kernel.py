#!/usr/bin/env python
# Kaggle GPU kernel: trains CodeForge-250M run #2 with watchdog, uploads
# checkpoint + metrics to a private Kaggle dataset after the session.
import os, sys, subprocess, shutil, glob

CF = "/kaggle/working/CodeForge-250M"
CKPT_DS = os.environ.get("KAGGLE_CHECKPOINT_DATASET", "yashbajpai2027/codeforge-ckpt")   # private dataset slug
DATA_DS = os.environ.get("KAGGLE_DATA_DATASET", "yashbajpai2027/codeforge-data")        # tokenized shards + tokenizer

os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=== [Kernel] boot: pulling latest code + data + checkpoint ===", flush=True)
subprocess.run(["pip", "install", "-q", "kaggle"])  # usually preinstalled

def kaggle_pull(slug, dest):
    os.makedirs(dest, exist_ok=True)
    r = subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", dest, "--unzip", "-o"])
    if r.returncode != 0:
        print(f"[Kernel] WARN: could not pull {slug}", flush=True)

# 1) code: prefer bundled copy (kernel push includes repo files), else git
if not os.path.exists(f"{CF}/training/train.py"):
    subprocess.run(["git", "clone", "-q",
                    "https://github.com/Yash1bajpai/CodeForge-250M.git", CF], check=False)

# 2) tokenized data + tokenizer
if not glob.glob(f"{CF}/data/tokenized/shard_*.pt"):
    kaggle_pull(DATA_DS, f"{CF}/data/tokenized")
# tokenizer lives in the same dataset under tokenizer/
os.makedirs(f"{CF}/data/tokenizer", exist_ok=True)
if not os.path.exists(f"{CF}/data/tokenizer/tokenizer.json"):
    for src in glob.glob(f"{CF}/data/tokenized/tokenizer/*"):
        shutil.copy(src, f"{CF}/data/tokenizer/")
    # remove non-shard files from tokenized dir
    for f in glob.glob(f"{CF}/data/tokenized/tokenizer"):
        shutil.rmtree(f, ignore_errors=True)

# 3) checkpoint (resume or fresh)
ckpt_dir = f"{CF}/checkpoints/CodeForge-250M"
os.makedirs(ckpt_dir, exist_ok=True)
if not os.path.exists(f"{ckpt_dir}/latest_checkpoint.pt"):
    kaggle_pull(CKPT_DS, "/tmp/ckpt_in")
    if os.path.exists("/tmp/ckpt_in/latest_checkpoint.pt"):
        shutil.copy("/tmp/ckpt_in/latest_checkpoint.pt", ckpt_dir)
        shutil.copy("/tmp/ckpt_in/metrics.json", CF) if os.path.exists("/tmp/ckpt_in/metrics.json") else None

# decide resume vs fresh
args = ["--resume"] if os.path.exists(f"{ckpt_dir}/latest_checkpoint.pt") else ["--from_scratch"]
print(f"=== [Kernel] training with args: {args} ===", flush=True)

# run training as a child so we can still upload after a crash
train_proc = subprocess.run([sys.executable, f"{CF}/training/train.py"] + args,
                            cwd=CF, capture_output=False)

# ---- always upload, even if training crashed: keep the evidence ----
print("=== [Kernel] uploading checkpoint + metrics ===", flush=True)
meta = f"""{{
  "title": "codeforge-ckpt",
  "id": "{CKPT_DS}",
  "licenses": [{{"name": "CC0-1.0"}}]
}}"""
os.makedirs("/tmp/ckpt_out", exist_ok=True)
with open("/tmp/ckpt_out/dataset-metadata.json", "w") as f:
    f.write(meta)
for src in [f"{ckpt_dir}/latest_checkpoint.pt", f"{CF}/metrics.json", f"{CF}/training.log"]:
    if os.path.exists(src):
        shutil.copy(src, "/tmp/ckpt_out/")
r = subprocess.run(["kaggle", "datasets", "version", "-p", "/tmp/ckpt_out",
                    "-m", "auto-checkpoint", "--dir-mode", "zip"])
print(f"=== [Kernel] upload rc={r.returncode}; train rc={train_proc.returncode} ===", flush=True)
