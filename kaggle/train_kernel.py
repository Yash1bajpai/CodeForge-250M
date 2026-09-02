#!/usr/bin/env python
# Kaggle GPU kernel: trains CodeForge-250M run #2 with watchdog, uploads
# checkpoint + metrics to a private Kaggle dataset after the session.
import os, sys, subprocess, shutil, glob

CF = "/kaggle/working/CodeForge-250M"
CKPT_DS = os.environ.get("KAGGLE_CHECKPOINT_DATASET", "yashbajpai2027/codeforge-ckpt")   # private dataset slug
DATA_DS = os.environ.get("KAGGLE_DATA_DATASET", "yashbajpai2027/codeforge-data")        # tokenized shards + tokenizer

os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

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

# 2) tokenized data + tokenizer: pull dataset to a STAGING dir, then place
# subdirs where train.py expects them (dataset zip contains tokenized/ + tokenizer/)
if not glob.glob(f"{CF}/data/tokenized/shard_*.pt") or not os.path.exists(f"{CF}/data/tokenizer/tokenizer.json"):
    kaggle_pull(DATA_DS, "/tmp/data_staging")
    import shutil as _sh2
    # case A: zip has top-level subdirs (tokenized/, tokenizer/)
    for sub in ["tokenized", "tokenizer"]:
        src = f"/tmp/data_staging/{sub}"
        dst = f"{CF}/data/{sub}"
        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            for f in glob.glob(f"{src}/*"):
                _sh2.copy(f, dst)
    # case B: zip was dir-mode-zip of data/ — files may sit at top level of staging
    if not glob.glob(f"{CF}/data/tokenized/shard_*.pt"):
        for f in glob.glob("/tmp/data_staging/shard_*.pt"):
            os.makedirs(f"{CF}/data/tokenized", exist_ok=True)
            _sh2.copy(f, f"{CF}/data/tokenized/")
    print(f"[Kernel] staged shards: {len(glob.glob(f'{CF}/data/tokenized/shard_*.pt'))} | "
          f"tokenizer: {os.path.exists(f'{CF}/data/tokenizer/tokenizer.json')}", flush=True)

# 3) checkpoint (resume or fresh) — RETRY: dataset version may still be
# processing right after a 3GB upload (caused silent 404 -> from-scratch retrain)
ckpt_dir = f"{CF}/checkpoints/CodeForge-250M"
os.makedirs(ckpt_dir, exist_ok=True)
prior_progress = 0
if not os.path.exists(f"{ckpt_dir}/latest_checkpoint.pt"):
    for attempt in range(5):
        ok = kaggle_pull(CKPT_DS, "/tmp/ckpt_in")
        if os.path.exists("/tmp/ckpt_in/latest_checkpoint.pt"):
            shutil.copy("/tmp/ckpt_in/latest_checkpoint.pt", ckpt_dir)
            break
        # does metrics say a checkpoint SHOULD exist?
        try:
            import json as _j
            m = [ _j.loads(l) for l in open("/tmp/ckpt_in/metrics.json") if l.strip() ] \
                if os.path.exists("/tmp/ckpt_in/metrics.json") else []
            trains = [e for e in m if e.get("event") == "TRAIN"]
            prior_progress = trains[-1]["step"] if trains else 0
        except Exception:
            prior_progress = 0
        if prior_progress == 0:
            break  # genuinely fresh run, no checkpoint expected
        print(f"[Kernel] ckpt pull attempt {attempt+1} failed (progress=step {prior_progress}); waiting 90s...", flush=True)
        import time as _t; _t.sleep(90)
    if prior_progress > 0 and not os.path.exists(f"{ckpt_dir}/latest_checkpoint.pt"):
        raise SystemExit(f"FATAL: training previously reached step {prior_progress} but checkpoint is unavailable after 5 pulls. Refusing to retrain from scratch (quota protection).")
if os.path.exists("/tmp/ckpt_in/metrics.json") and not os.path.exists(f"{CF}/metrics.json"):
    shutil.copy("/tmp/ckpt_in/metrics.json", CF)

# decide resume vs fresh
args = ["--resume"] if os.path.exists(f"{ckpt_dir}/latest_checkpoint.pt") else ["--from_scratch"]
print(f"=== [Kernel] training with args: {args} ===", flush=True)

# GPU count: Kaggle T4 x2 exposes 2 CUDA devices -> torchrun DDP across both.
n_gpus = 0
try:
    import torch
    n_gpus = torch.cuda.device_count()
except Exception:
    pass
if n_gpus >= 2:
    print(f"=== [Kernel] launching DDP on {n_gpus} GPUs ===", flush=True)
    train_proc = subprocess.run(["torchrun", f"--nproc_per_node={n_gpus}",
                                 f"{CF}/training/train.py"] + args,
                                cwd=CF, capture_output=False)
else:
    print("=== [Kernel] single-GPU mode ===", flush=True)
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
