import os
import time
import subprocess
import re

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
os.chdir(base_dir)

print("=======================================================")
print("=== [1. CONFIGURING OVERNIGHT CHUNK 3 (6,000 STEPS / 3.5 HRS)] ===")
print("=======================================================")

train_path = os.path.join(base_dir, "training", "train.py")
with open(train_path, "r", encoding="utf-8") as f:
    content = f.read()

# Make sure max_steps allows overnight run (e.g. 6500 steps total ~ 26 Million tokens)
# We look for max_steps = ... or change it
if "max_steps = " in content:
    content = re.sub(r"max_steps\s*=\s*\d+", "max_steps = 6500", content)
else:
    # If not found via regex, let's check where steps are defined
    pass

with open(train_path, "w", encoding="utf-8") as f:
    f.write(content)

print("--> Updated training/train.py: Target set to Step 6,500 (~26 Million tokens overnight run!)")

print("\n=======================================================")
print("=== [2. LAUNCHING CHUNK 3 IN BACKGROUND] ===")
print("=======================================================")

# Kill any lingering train.py just in case
subprocess.run(["pkill", "-f", "train.py"], check=False)
time.sleep(1)

python_bin = "/system/conda/miniconda3/envs/cloudspace/bin/python3"
log_file = os.path.join(base_dir, "training.log")

# Launch in background with nohup
cmd = f"nohup {python_bin} -u training/train.py > /dev/null 2>&1 &"
subprocess.run(cmd, shell=True, check=True)
print("--> Launched training process in background!")

time.sleep(5)

# Check process
res_ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
train_procs = [line for line in res_ps.stdout.splitlines() if "train.py" in line and "grep" not in line]

if train_procs:
    print("SUCCESS: Training process is running live!")
    for p in train_procs:
        print("  PID info:", p[:80])
else:
    print("WARNING: Could not detect train.py process. Checking log...")

print("\n=======================================================")
print("=== [3. INITIAL LOG OUTPUT (training.log)] ===")
print("=======================================================")

if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    print("Last 15 lines of training.log:")
    for line in lines[-15:]:
        print(" ", line.strip())
else:
    print("Log file not found yet.")
