#!/usr/bin/env python
# Kaggle GPU kernel: trains CodeForge-250M run #2 with watchdog, uploads
# checkpoint + metrics to a private Kaggle dataset after the session.
import os, sys, subprocess, shutil, glob, time

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

# 3) checkpoint (resume or fresh) — QUOTA-CRITICAL: never silently retrain.
# A session may start from scratch ONLY if the ckpt dataset verifiably holds
# no checkpoint AND no recorded training progress. Any ambiguity = FATAL.
ckpt_dir = f"{CF}/checkpoints/CodeForge-250M"
os.makedirs(ckpt_dir, exist_ok=True)

def pull_ckpt_dataset(dest):
    r = subprocess.run(["kaggle", "datasets", "download", "-d", CKPT_DS,
                        "-p", dest, "--unzip", "-o"],
                       capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")

if not os.path.exists(f"{ckpt_dir}/latest_checkpoint.pt"):
    pull_rc, pull_out = 1, ""
    for attempt in range(8):
        pull_rc, pull_out = pull_ckpt_dataset("/tmp/ckpt_in")
        if pull_rc == 0:
            break
        print(f"[Kernel] ckpt dataset pull failed (attempt {attempt+1}/8, rc={pull_rc}); "
              f"retry in 90s. tail: {pull_out[-200:]}", flush=True)
        time.sleep(90)
    if os.path.exists("/tmp/ckpt_in/latest_checkpoint.pt"):
        shutil.copy("/tmp/ckpt_in/latest_checkpoint.pt", ckpt_dir)
        print("[Kernel] checkpoint restored from dataset — resuming", flush=True)
    elif pull_rc == 0:
        # Full dataset in hand, no checkpoint file inside: fresh ONLY if no history.
        prior = 0
        try:
            import json as _j
            m = [_j.loads(l) for l in open("/tmp/ckpt_in/metrics.json") if l.strip()]
            prior = max((e.get("step", 0) for e in m if e.get("event") == "TRAIN"), default=0)
        except FileNotFoundError:
            prior = 0
        if prior > 0:
            raise SystemExit(
                f"FATAL: metrics show step {prior} but dataset has no latest_checkpoint.pt "
                f"(bad upload). Refusing to retrain from scratch.")
        print("[Kernel] no checkpoint and no training history — fresh run", flush=True)
    else:
        raise SystemExit(
            f"FATAL: cannot pull ckpt dataset after 8 attempts (rc={pull_rc}). "
            f"Refusing to retrain from scratch. tail: {pull_out[-200:]}")

if os.path.exists("/tmp/ckpt_in/metrics.json") and not os.path.exists(f"{CF}/metrics.json"):
    shutil.copy("/tmp/ckpt_in/metrics.json", CF)

# decide resume vs fresh
args = ["--resume"] if os.path.exists(f"{ckpt_dir}/latest_checkpoint.pt") else ["--from_scratch"]
print(f"=== [Kernel] training with args: {args} ===", flush=True)

# ALWAYS single-GPU: Kaggle's NvidiaTeslaT4 shape exposes 2 devices, but T4-x2 DDP
# dies from NCCL rank desync at ~step 5 (rank0 SIGTERM / rank1 SIGABRT). Pin to GPU 0.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
try:
    gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except Exception:
    gpu_info = "unknown"
print(f"=== [Kernel] GPU: {gpu_info} ===", flush=True)
print("=== [Kernel] single-GPU mode (forced, GPU 0 of available) ===", flush=True)
# Kaggle hard-kills GPU sessions at 9h; stop at 8.5h so save+upload always run.
train_proc = subprocess.run([sys.executable, f"{CF}/training/train.py"] + args +
                            ["--max_hours", "8.5"],
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
