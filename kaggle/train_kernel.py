#!/usr/bin/env python
# Kaggle GPU kernel (v25): trains CodeForge-250M run #2.
# Datasets are attached as kernel data sources (instant read-only mounts at
# /kaggle/input/...) — no 3GB API downloads, no 404 processing races.
# Training exit code is propagated: crashes mark the kernel ERROR so the
# supervisor alerts instead of silently crash-looping.
import os, sys, subprocess, shutil, glob, time, json

CF = "/kaggle/working/CodeForge-250M"
CKPT_DS = "yashbajpai2027/codeforge-ckpt"
DATA_MOUNT = "/kaggle/input/codeforge-data"
CKPT_MOUNT = "/kaggle/input/codeforge-ckpt"

os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

print("=== [Kernel] boot (v25) ===", flush=True)
subprocess.run(["pip", "install", "-q", "kaggle"])

def sh(cmd, **kw):
    return subprocess.run(cmd, **kw)

def symlink_into(src_dir, dst_dir):
    """Link every entry of src_dir into dst_dir (mounts are read-only)."""
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for entry in glob.glob(os.path.join(src_dir, "*")):
        dst = os.path.join(dst_dir, os.path.basename(entry))
        if not os.path.exists(dst):
            try:
                os.symlink(entry, dst)
            except OSError:
                shutil.copy(entry, dst)
        n += 1
    return n

# 1) code
if not os.path.exists(f"{CF}/training/train.py"):
    sh(["git", "clone", "-q", "https://github.com/Yash1bajpai/CodeForge-250M.git", CF], check=False)
if not os.path.exists(f"{CF}/training/train.py"):
    raise SystemExit("FATAL: could not obtain training code (git clone failed)")

# 2) data + tokenizer: kernel-attached mount, symlinked into the project tree
staged = 0
for cand in [DATA_MOUNT,
             os.path.join(DATA_MOUNT, "data")]:
    if glob.glob(os.path.join(cand, "tokenized", "shard_*.pt")):
        staged += symlink_into(os.path.join(cand, "tokenized"), f"{CF}/data/tokenized")
        staged += symlink_into(os.path.join(cand, "tokenizer"), f"{CF}/data/tokenizer")
        break
if staged == 0:
    # fallback: legacy API pull (dataset not attached as source)
    print("[Kernel] no data mount — falling back to API pull", flush=True)
    sh(["kaggle", "datasets", "download", "-d", "yashbajpai2027/codeforge-data",
        "-p", "/tmp/data_staging", "--unzip", "-o"])
    for sub in ["tokenized", "tokenizer"]:
        src = f"/tmp/data_staging/{sub}"
        if os.path.isdir(src):
            staged += symlink_into(src, f"{CF}/data/{sub}")
    if glob.glob("/tmp/data_staging/shard_*.pt"):
        os.makedirs(f"{CF}/data/tokenized", exist_ok=True)
        staged += symlink_into("/tmp/data_staging", f"{CF}/data/tokenized")
print(f"[Kernel] staged: {staged} data entries | "
      f"shards: {len(glob.glob(f'{CF}/data/tokenized/shard_*.pt'))} | "
      f"tokenizer: {os.path.exists(f'{CF}/data/tokenizer/tokenizer.json')}", flush=True)

# 3) checkpoint + metrics: mount first, API best-effort merge
ckpt_dir = f"{CF}/checkpoints/CodeForge-250M"
os.makedirs(ckpt_dir, exist_ok=True)
metrics_path = f"{CF}/metrics.json"

if os.path.exists(f"{CKPT_MOUNT}/latest_checkpoint.pt"):
    shutil.copy(f"{CKPT_MOUNT}/latest_checkpoint.pt", f"{ckpt_dir}/latest_checkpoint.pt")
    print("[Kernel] checkpoint restored from kernel-attached mount", flush=True)

# metrics: prefer the freshest of (mount, API). The mount can lag one version
# if the newest upload is still processing server-side.
metrics_candidates = []
for mpath in [f"{CKPT_MOUNT}/metrics.json"]:
    if os.path.exists(mpath):
        metrics_candidates.append(open(mpath).read())
try:
    r = sh(["kaggle", "datasets", "download", "-d", CKPT_DS, "-f", "metrics.json",
            "-p", "/tmp/mapi", "--unzip", "-o"], capture_output=True, text=True, timeout=180)
    if os.path.exists("/tmp/mapi/metrics.json"):
        metrics_candidates.append(open("/tmp/mapi/metrics.json").read())
except Exception as e:
    print(f"[Kernel] metrics API pull skipped: {e}", flush=True)
if metrics_candidates and not os.path.exists(metrics_path):
    best = max(metrics_candidates, key=len)  # most complete history wins
    open(metrics_path, "w").write(best)

def max_step_in(text):
    try:
        entries = [json.loads(l) for l in text.splitlines() if l.strip()]
        return max((e.get("step", 0) for e in entries
                    if e.get("event") in ("TRAIN", "FINAL", "CKPT", "CRASH")), default=0)
    except Exception:
        return 0

prior_step = 0
if os.path.exists(metrics_path):
    prior_step = max_step_in(open(metrics_path).read())
print(f"[Kernel] prior recorded progress: step {prior_step}", flush=True)

# resume vs fresh — QUOTA-CRITICAL: never silently retrain from scratch
if not os.path.exists(f"{ckpt_dir}/latest_checkpoint.pt"):
    if prior_step == 0:
        print("[Kernel] no checkpoint and no history — fresh run", flush=True)
    else:
        ok = False
        for att in range(4):
            r = sh(["kaggle", "datasets", "download", "-d", CKPT_DS,
                    "-p", "/tmp/ckpt_api", "--unzip", "-o"],
                   capture_output=True, text=True, timeout=600)
            if os.path.exists("/tmp/ckpt_api/latest_checkpoint.pt"):
                shutil.move("/tmp/ckpt_api/latest_checkpoint.pt", f"{ckpt_dir}/latest_checkpoint.pt")
                ok = True
                break
            print(f"[Kernel] ckpt API pull attempt {att+1}/4 failed; retry in 120s", flush=True)
            time.sleep(120)
        if not ok:
            raise SystemExit(
                f"FATAL: prior training reached step {prior_step} but checkpoint "
                f"is unavailable. Refusing to retrain from scratch (quota protection).")

args = ["--resume"] if os.path.exists(f"{ckpt_dir}/latest_checkpoint.pt") else ["--from_scratch"]

# ALWAYS single-GPU (T4x2 DDP dies from NCCL rank desync at ~step 5).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
try:
    gpu_info = sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                  capture_output=True, text=True).stdout.strip()
except Exception:
    gpu_info = "unknown"
print(f"=== [Kernel] GPU: {gpu_info} | single-GPU mode (forced) | args: {args} ===", flush=True)
# Kaggle hard-kills GPU sessions; stop at 8.5h so save+upload always run.
train_proc = sh([sys.executable, f"{CF}/training/train.py"] + args + ["--max_hours", "8.5"],
                cwd=CF)
print(f"=== [Kernel] train rc={train_proc.returncode} ===", flush=True)

# ---- always upload diagnostics, even if training crashed: keep the evidence ----
print("=== [Kernel] uploading checkpoint + metrics ===", flush=True)
os.makedirs("/tmp/ckpt_out", exist_ok=True)
with open("/tmp/ckpt_out/dataset-metadata.json", "w") as f:
    f.write(json.dumps({"title": "codeforge-ckpt", "id": CKPT_DS,
                        "licenses": [{"name": "CC0-1.0"}]}))
for src in [f"{ckpt_dir}/latest_checkpoint.pt", metrics_path, f"{CF}/training.log"]:
    if os.path.exists(src):
        shutil.copy(src, "/tmp/ckpt_out/")

if os.path.exists("/tmp/ckpt_out/metrics.json"):
    step_now = max_step_in(open("/tmp/ckpt_out/metrics.json").read())
    with open("/tmp/ckpt_out/metrics.json", "a") as f:
        f.write(json.dumps({"event": "TRAIN_EXIT", "step": step_now,
                            "rc": train_proc.returncode, "ts": time.time()}) + "\n")
    with open("/tmp/ckpt_out/step.json", "w") as f:
        json.dump({"step": step_now}, f)

r = sh(["kaggle", "datasets", "version", "-p", "/tmp/ckpt_out",
        "-m", f"auto-checkpoint", "--dir-mode", "zip"])
print(f"=== [Kernel] upload rc={r.returncode} ===", flush=True)

# propagate training failure: kernel -> ERROR -> supervisor ALERTs (no silent loops)
sys.exit(train_proc.returncode if train_proc.returncode != 0 else (0 if r.returncode == 0 else 1))
