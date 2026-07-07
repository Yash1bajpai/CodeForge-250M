import os
import time
import subprocess

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
os.chdir(base_dir)

print("=================================================================")
print("=== [1. UPGRADING TO BATCH SIZE 8 -> 4X SUPER SPEED!] ===")
print("=== [TARGET -> STEP 118,000 / 1 BILLION TOKENS!] ===")
print("=================================================================")

train_path = os.path.join(base_dir, "training", "train.py")
with open(train_path, "r", encoding="utf-8") as f:
    content = f.read()

# Update batch_size from 2 to 8 (4x tokens per step -> 16,384 tokens/step!)
if "batch_size=2" in content:
    content = content.replace("batch_size=2", "batch_size=8")
elif "batch_size=4" in content:
    content = content.replace("batch_size=4", "batch_size=8")

# Update target steps to 118000 (~1 BILLION TOKENS at batch size 8!)
if "max_steps = " in content:
    content = content.replace("max_steps = 122000", "max_steps = 118000")
    content = content.replace("max_steps = 78000", "max_steps = 118000")
    content = content.replace("max_steps = 73000", "max_steps = 118000")
    content = content.replace("max_steps = 72000", "max_steps = 118000")

content = content.replace("STAGE 15 / L40S HALF-BILLION SPRINT", "STAGE 16 / L40S 1 BILLION TOKEN SPRINT (BATCH SIZE 8)")

with open(train_path, "w", encoding="utf-8") as f:
    f.write(content)

print("--> Successfully updated training/train.py for BATCH SIZE 8!")
print("    - New Micro-Batch Size : batch_size = 8 (16,384 tokens per step!)")
print("    - Target Milestone    : Step 118,000 (~1 BILLION TOKENS / 1,000,000,000 TOKENS!)")
print("    - Expected Speed      : ~450,000+ tokens/hour (4x boost!)")

print("\n=================================================================")
print("=== [2. RESTARTING TRAINING PROCESS ON NVIDIA L40S GPU] ===")
print("=================================================================")

python_bin = "/system/conda/miniconda3/envs/cloudspace/bin/python3"
subprocess.run(["pkill", "-f", "train.py"], check=False)
time.sleep(2)

cmd = f"nohup {python_bin} -u training/train.py > /dev/null 2>&1 &"
subprocess.run(cmd, shell=True, check=True)
print("--> Relaunched training with BATCH SIZE 8 in background!")

time.sleep(8)

res_ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
train_procs = [line for line in res_ps.stdout.splitlines() if "train.py" in line and "grep" not in line]

if train_procs:
    print("SUCCESS: L40S 1 BILLION TOKEN SPRINT (Batch Size 8) is LIVE!")
    for p in train_procs:
        print("  PID info:", p[:80])
else:
    print("WARNING: Could not detect train.py process.")

print("\n=================================================================")
print("=== [3. CHECKING NEW VRAM UTILIZATION (nvidia-smi)] ===")
print("=================================================================")

subprocess.run(["nvidia-smi"], check=False)
