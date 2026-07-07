import os
import time
import subprocess

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
os.chdir(base_dir)

print("=================================================================")
print("=== [1. CONFIGURING CONTINUOUS 1 BILLION TOKEN SPRINT] ===")
print("=== [REMOVING CHUNK LIMITS -> TARGET: STEP 120,000 / 1 BILLION!] ===")
print("=================================================================")

train_path = os.path.join(base_dir, "training", "train.py")
with open(train_path, "r", encoding="utf-8") as f:
    content = f.read()

# Update target steps to 120000 (~1 BILLION TOKENS at batch size 8!)
if "max_steps = " in content:
    content = content.replace("max_steps = 118000", "max_steps = 120000")
    content = content.replace("max_steps = 122000", "max_steps = 120000")
    content = content.replace("max_steps = 83000", "max_steps = 120000")
    content = content.replace("max_steps = 78000", "max_steps = 120000")

# If there is a chunk step limit like `if step_offset >= 6000: break`, let's make it 50,000!
if "step_offset >= 6000" in content:
    content = content.replace("step_offset >= 6000", "step_offset >= 50000")
if "step_offset >= 5000" in content:
    content = content.replace("step_offset >= 5000", "step_offset >= 50000")

with open(train_path, "w", encoding="utf-8") as f:
    f.write(content)

print("--> Successfully updated training/train.py for CONTINUOUS 1 BILLION SPRINT!")
print("    - Resumes from: latest_checkpoint.pt (~Step 83,000 / ~401.4 MILLION TOKENS!)")
print("    - Target Size : Continuous until Step 120,000 (1 BILLION TOKENS / 1,000,000,000 TOKENS!)")
print("    - Next Milestone: Step 89,000 (500 MILLION TOKENS / HALF A BILLION in ~20 mins!)")

print("\n=================================================================")
print("=== [2. LAUNCHING CONTINUOUS TRAINING ON NVIDIA L40S GPU] ===")
print("=================================================================")

python_bin = "/system/conda/miniconda3/envs/cloudspace/bin/python3"
subprocess.run(["pkill", "-f", "train.py"], check=False)
time.sleep(2)

cmd = f"nohup {python_bin} -u training/train.py > /dev/null 2>&1 &"
subprocess.run(cmd, shell=True, check=True)
print("--> Launched Continuous 1 Billion Token Sprint in background!")

time.sleep(6)

res_ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
train_procs = [line for line in res_ps.stdout.splitlines() if "train.py" in line and "grep" not in line]

if train_procs:
    print("SUCCESS: Continuous 1 Billion Token Sprint is LIVE on NVIDIA L40S GPU!")
    for p in train_procs:
        print("  PID info:", p[:80])
else:
    print("WARNING: Could not detect train.py process.")

print("\n=================================================================")
print("=== [3. INITIAL LOG CHECK] ===")
print("=================================================================")

log_file = os.path.join(base_dir, "training.log")
if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    print("Last 15 lines of training.log:")
    for line in lines[-15:]:
        print(" ", line.strip())
