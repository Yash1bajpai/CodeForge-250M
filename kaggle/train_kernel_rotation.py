#!/usr/bin/env python
# Kaggle GPU kernel (ROTATION v1): trains CodeForge-250M run #2 on ANY account.
# Difference from train_kernel.py: the checkpoint flows through HuggingFace Hub
# (Yash1bajpai/CodeForge-250M-rotation) instead of account-local Kaggle
# datasets, so account B/C kernels resume the exact state account A pushed.
# The training data still mounts from the owner's Kaggle dataset (read-only
# attach — friend accounts can mount other users' public... they CAN'T; so
# data is pulled from HF at boot when not mounted).
# HF_TOKEN is injected via kernel push metadata (a Kaggle "secret" variable
# placeholder that the pusher replaces locally before pushing — the token
# never appears in any repo or log).
import os, sys, subprocess, shutil, glob, time, json

CF = "/kaggle/working/CodeForge-250M"
HF_REPO = "Yash1bajpai/CodeForge-250M-rotation"
DATA_MOUNT = "/kaggle/input/codeforge-data"
# --- the pusher replaces this line with the real token at push time ---
HF_TOKEN = "__HF_TOKEN_PLACEHOLDER__"

os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_TOKEN"] = HF_TOKEN

print("=== [Kernel] boot (rotation v1) ===", flush=True)
subprocess.run(["pip", "install", "-q", "huggingface_hub"], check=False)

def sh(cmd, **kw):
    return subprocess.run(cmd, **kw)

def hf_download(path_in_repo, dest):
    """Pull a file from the rotation repo. MISSING file = legit fresh start;
    any OTHER error (auth, rate limit, network) = HARD EXIT — never fall back
    to --from_scratch on a transient error and overwrite the shared checkpoint."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    last_err = None
    for att in range(4):
        try:
            p = hf_hub_download(repo_id=HF_REPO, filename=path_in_repo,
                                token=HF_TOKEN, local_dir=os.path.dirname(dest))
            return p
        except (EntryNotFoundError, RepositoryNotFoundError):
            print(f"[Kernel] {path_in_repo} not in rotation repo (legit first run)", flush=True)
            return None
        except Exception as e:
            last_err = e
            print(f"[Kernel] hf pull {path_in_repo} attempt {att+1}/4 failed: {e}", flush=True)
            time.sleep(30)
    raise SystemExit(f"FATAL: cannot pull {path_in_repo} from HF after 4 attempts "
                     f"({last_err}). Refusing to start from scratch and destroy the "
                     f"shared checkpoint.")

def hf_upload(path, path_in_repo, msg):
    """Upload with rewind guard: if the repo's current checkpoint is AHEAD of
    ours (another account ran further), refuse to overwrite — last-writer-wins
    would silently roll the run back."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    try:
        api = HfApi(token=HF_TOKEN)
        if path_in_repo == "latest_checkpoint.pt":
            _tmpd = None
            try:
                import torch as _t
                import tempfile as _tf
                _tmpd = _tf.mkdtemp(prefix="rewind_")
                remote = api.hf_hub_download(repo_id=HF_REPO, filename="latest_checkpoint.pt",
                                             token=HF_TOKEN, cache_dir=_tmpd)
                ours = _t.load(path, map_location="cpu")
                try:
                    theirs = _t.load(remote, map_location="cpu")
                except Exception:
                    theirs = {"step": None}  # remote unreadable -> do not overwrite blindly
                if theirs.get("step") is not None and (ours.get("step") or 0) < theirs.get("step", 0):
                    print(f"[Kernel] REWIND GUARD: remote step {theirs['step']} > ours "
                          f"{ours.get('step')} — NOT overwriting", flush=True)
                    return True  # not an error; the newer state is already there
                del ours, theirs
            except (EntryNotFoundError, RepositoryNotFoundError, FileNotFoundError):
                pass  # first push, nothing to guard
            finally:
                # clean the multi-GB guard download immediately
                import shutil as _sh
                if _tmpd:
                    _sh.rmtree(_tmpd, ignore_errors=True)
        api.upload_file(path_or_fileobj=path, path_in_repo=path_in_repo,
                        repo_id=HF_REPO, repo_type="model", commit_message=msg)
        return True
    except Exception as e:
        print(f"[Kernel] hf push {path_in_repo} failed: {e}", flush=True)
        return False

# 1) training code: git clone (public repo)
if not os.path.exists(f"{CF}/training/train.py"):
    sh(["git", "clone", "-q", "https://github.com/Yash1bajpai/CodeForge-250M.git", CF], check=False)
if not os.path.exists(f"{CF}/training/train.py"):
    raise SystemExit("FATAL: could not obtain training code (git clone failed)")

# 2) training data: mount first (owner account), HF fallback (friend accounts).
#    codeforge-data is PUBLIC (owner set it so friend kernels can attach it),
#    745 shards ~1.4GB zipped. HF fallback pulls the same shards from the
#    rotation repo's data/ prefix (staged by the watchdog).
staged = 0
for cand in [DATA_MOUNT, os.path.join(DATA_MOUNT, "data")]:
    if glob.glob(os.path.join(cand, "tokenized", "shard_*.pt")):
        os.makedirs(f"{CF}/data/tokenized", exist_ok=True)
        os.makedirs(f"{CF}/data/tokenizer", exist_ok=True)
        for sub in ("tokenized", "tokenizer"):
            src = os.path.join(cand, sub)
            for entry in glob.glob(os.path.join(src, "*")):
                dst = os.path.join(f"{CF}/data", sub, os.path.basename(entry))
                if not os.path.exists(dst):
                    try:
                        os.symlink(entry, dst)
                    except OSError:
                        shutil.copy(entry, dst)
                staged += 1
        break
print(f"[Kernel] shards staged from mount: {staged} | "
      f"{len(glob.glob(f'{CF}/data/tokenized/shard_*.pt'))} visible", flush=True)

if staged == 0:
    # friend account: data comes from the HF rotation repo (data/ prefix)
    print("[Kernel] no Kaggle data mount — pulling shards from HF rotation repo", flush=True)
    from huggingface_hub import list_repo_files, hf_hub_download
    EXPECTED_SHARDS = None  # read manifest if present
    files = [f for f in list_repo_files(repo_id=HF_REPO, token=HF_TOKEN)
             if f.startswith("data/tokenized/") and f.endswith(".pt")]
    # shard-count contract: a short staging silently changes the seed-42
    # permutation and the val split, breaking single-pass resume. Fail hard.
    try:
        mani = hf_hub_download(repo_id=HF_REPO, filename="data/manifest.json",
                               token=HF_TOKEN)
        EXPECTED_SHARDS = json.load(open(mani)).get("n_shards")
    except Exception:
        pass
    if EXPECTED_SHARDS is not None and len(files) != EXPECTED_SHARDS:
        raise SystemExit(f"FATAL: rotation repo has {len(files)} shards, manifest says "
                         f"{EXPECTED_SHARDS}. Partial staging would corrupt the "
                         f"permutation/val-split. Do not train.")
    print(f"[Kernel] {len(files)} shards in rotation repo "
          f"(manifest: {EXPECTED_SHARDS})", flush=True)
    if not files:
        raise SystemExit("FATAL: no shards on Kaggle mount or HF rotation repo")
    os.makedirs(f"{CF}/data/tokenized", exist_ok=True)
    for i, f in enumerate(files):
        # local_dir preserves the repo path: tokenized_hf/data/tokenized/shard_X.pt
        p = None
        for att in range(4):
            try:
                p = hf_hub_download(repo_id=HF_REPO, filename=f, token=HF_TOKEN,
                                    local_dir=f"{CF}/data/tokenized_hf")
                break
            except Exception as e:
                print(f"[Kernel] shard pull {f} attempt {att+1}/4 failed: {e}", flush=True)
                time.sleep(20)
        if p is None:
            raise SystemExit(f"FATAL: shard {f} would not download after 4 attempts")
        dst = f"{CF}/data/tokenized/{os.path.basename(f)}"
        if not os.path.exists(dst):
            try:
                os.symlink(p, dst)
            except OSError:
                shutil.copy(p, dst)
        if (i + 1) % 100 == 0:
            print(f"[Kernel] staged {i+1}/{len(files)} shards", flush=True)
    # tokenizer: pull every file under data/tokenizer/ (json + config siblings)
    tok_files = [f for f in list_repo_files(repo_id=HF_REPO, token=HF_TOKEN)
                 if f.startswith("data/tokenizer/") and not f.endswith("/") and len(f) > len("data/tokenizer/")]
    os.makedirs(f"{CF}/data/tokenizer", exist_ok=True)
    for tf in tok_files:
        p = hf_hub_download(repo_id=HF_REPO, filename=tf, token=HF_TOKEN,
                            local_dir=f"{CF}/data/tokenizer_hf")
        dst = f"{CF}/data/tokenizer/{os.path.basename(tf)}"
        if not os.path.exists(dst):
            shutil.copy(p, dst)
    print(f"[Kernel] tokenizer staged: {len(tok_files)} files", flush=True)

# 3) checkpoint: ALWAYS from HF (the single source of truth across accounts)
ckpt_dir = f"{CF}/checkpoints/CodeForge-250M"
os.makedirs(ckpt_dir, exist_ok=True)
hf_ck = hf_download("latest_checkpoint.pt", f"{ckpt_dir}/latest_checkpoint.pt")
if hf_ck is None:
    print("[Kernel] no HF checkpoint — starting FROM SCRATCH (fresh run)", flush=True)
    args = ["--from_scratch"]
else:
    print(f"[Kernel] HF checkpoint restored: {hf_ck}", flush=True)
    args = ["--resume"]

# 4) telemetry path: the in-kernel train.py pushes to Kaggle datasets using
#    the account's own kaggle CLI... but friend accounts have no codeforge
#    telemetry datasets. Reuse the OWNER's telemetry dataset is impossible
#    (write permission). Instead: train.py telemetry is disabled here; the
#    training.log + metrics.json are pushed to the HF rotation repo directly
#    by this kernel at session end (and every 30 min via background thread).
os.environ["CF_TELEMETRY_DISABLED"] = "1"  # train.py reads this (v27 change)

# ALWAYS single-GPU (T4x2 DDP desync). P100 unusable (PyTorch dropped sm_60).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
try:
    gpu_info = sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                  capture_output=True, text=True).stdout.strip()
except Exception:
    gpu_info = "unknown"
print(f"=== [Kernel] GPU: {gpu_info} | single-GPU | args: {args} ===", flush=True)

# live telemetry thread: push metrics.json + step.json + log tail to HF every 15 min
_TELE_STOP = []
def _telemeter():
    import threading
    def _loop():
        while not _TELE_STOP:
            time.sleep(900)
            try:
                m = f"{CF}/metrics.json"
                if os.path.exists(m):
                    hf_upload(m, "metrics.json", "telemetry")
                s = f"{CF}/step.json"
                if os.path.exists(m):
                    import json as _j
                    step = 0
                    try:
                        with open(m) as f:
                            step = max((_j.loads(l).get("step", 0)
                                        for l in f if l.strip()), default=0)
                    except Exception:
                        pass
                    with open(s, "w") as f:
                        _j.dump({"step": step, "ts": time.time()}, f)
                    hf_upload(s, "step.json", "telemetry")
                lg = f"{CF}/training.log"
                if os.path.exists(lg):
                    with open(lg, "r", errors="replace") as f:
                        tail = f.readlines()[-400:]
                    with open("/tmp/training_tail.log", "w") as f:
                        f.writelines(tail)
                    hf_upload("/tmp/training_tail.log", "training_tail.log", "telemetry")
            except Exception as e:
                print(f"[Kernel] telemetry push failed (ignored): {e}", flush=True)
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t

telem = _telemeter()

# Kaggle hard-kills GPU sessions; stop at 8.5h so save+upload always run.
train_proc = sh([sys.executable, f"{CF}/training/train.py"] + args + ["--max_hours", "8.5"],
                cwd=CF)
print(f"=== [Kernel] train rc={train_proc.returncode} ===", flush=True)
_TELE_STOP.append(1)  # stop telemetry thread

# ---- always upload evidence + checkpoint to HF, even on crash ----
print("=== [Kernel] uploading checkpoint + diagnostics to HF ===", flush=True)
if os.path.exists(f"{ckpt_dir}/latest_checkpoint.pt"):
    hf_upload(f"{ckpt_dir}/latest_checkpoint.pt", "latest_checkpoint.pt",
              f"auto-checkpoint (rotation)")
if os.path.exists(f"{CF}/metrics.json"):
    hf_upload(f"{CF}/metrics.json", "metrics.json", "metrics")
if os.path.exists(f"{CF}/training.log"):
    hf_upload(f"{CF}/training.log", "training.log", "log")

rc = train_proc.returncode
sys.exit(rc if rc != 0 else 0)
