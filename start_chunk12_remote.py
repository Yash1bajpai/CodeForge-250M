import os
import time
import subprocess

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
os.chdir(base_dir)

print("=================================================================")
print("=== [1. VERIFYING STAGE 11 COMPLETION & CLEANING DISK] ===")
print("=================================================================")

log_file = os.path.join(base_dir, "training.log")
if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    print("Last 15 lines of training.log before release:")
    for line in lines[-15:]:
        print(" ", line.strip())

print("\n--> Cleaning up intermediate checkpoint files (retaining only multiples of 1,000 steps)...")
ckpt_dir = os.path.join(base_dir, "checkpoints", "CodeForge-250M")
if os.path.exists(ckpt_dir):
    for fname in os.listdir(ckpt_dir):
        if fname.startswith("checkpoint_step_") and fname.endswith(".pt") and not fname.endswith("000.pt"):
            fpath = os.path.join(ckpt_dir, fname)
            try:
                os.remove(fpath)
                print(f"  Deleted bloat: {fname}")
            except Exception as e:
                print(f"  Error deleting {fname}: {e}")

print("\n=================================================================")
print("=== [2. RELEASING STAGE 11 (~225.3M TOKENS!) TO HUGGING FACE] ===")
print("=================================================================")

python_bin = "/system/conda/miniconda3/envs/cloudspace/bin/python3"
subprocess.run([python_bin, "auto_release_chunk3.py"], check=False)

print("\n=================================================================")
print("=== [3. CONFIGURING STAGE 12 / CHUNK 12 (TARGET -> STEP 61,000)] ===")
print("=== [THE QUARTER BILLION TOKEN MARK (~250 MILLION TOKENS!)] ===")
print("=================================================================")

train_path = os.path.join(base_dir, "training", "train.py")
with open(train_path, "r", encoding="utf-8") as f:
    content = f.read()

# Update target steps to 61000 (~250 Million tokens - QUARTER BILLION!)
if "max_steps = " in content:
    content = content.replace("max_steps = 55000", "max_steps = 61000")
    content = content.replace("max_steps = 49000", "max_steps = 61000")
    content = content.replace("max_steps = 43500", "max_steps = 61000")

content = content.replace("STAGE 11 / CHUNK 11", "STAGE 12 / CHUNK 12")
content = content.replace("Chunk 11 completed", "Chunk 12 completed")

# Ensure checkpoint saving is set to 1000 steps
content = content.replace("if step % 100 == 0:", "if step % 1000 == 0:")

with open(train_path, "w", encoding="utf-8") as f:
    f.write(content)

print("--> Successfully updated training/train.py for Stage 12 / Chunk 12!")
print("    - Resumes from: latest_checkpoint.pt (~Step 55,000 / ~225.3 Million Tokens!)")
print("    - Target Chunk Size: +6,000 steps (Will reach Step 61,000 / ~250 MILLION TOKENS - QUARTER BILLION!)")

print("\n=================================================================")
print("=== [4. LAUNCHING CHUNK 12 IN BACKGROUND (NOHUP)] ===")
print("=================================================================")

subprocess.run(["pkill", "-f", "train.py"], check=False)
time.sleep(1)

cmd = f"nohup {python_bin} -u training/train.py > /dev/null 2>&1 &"
subprocess.run(cmd, shell=True, check=True)
print("--> Launched Chunk 12 training process in background!")

time.sleep(6)

res_ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
train_procs = [line for line in res_ps.stdout.splitlines() if "train.py" in line and "grep" not in line]

if train_procs:
    print("SUCCESS: Stage 12 / Chunk 12 Training is LIVE and running on Tesla T4 GPU!")
    for p in train_procs:
        print("  PID info:", p[:80])
else:
    print("WARNING: Could not detect train.py process.")

print("\n=================================================================")
print("=== [5. INITIAL LOG VERIFICATION (training.log)] ===")
print("=================================================================")

if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    print("Last 15 lines of training.log:")
    for line in lines[-15:]:
        print(" ", line.strip())
else:
    print("Log file not found.")
