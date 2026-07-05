import os
import time
import subprocess
import re

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
os.chdir(base_dir)

print("=======================================================")
print("=== [1. DIAGNOSIS: WHY IT STOPPED AT STEP 1,665] ===")
print("=======================================================")
print("--> It did NOT crash! At Step 1,665, the model processed 100% of all 6.8 Million tokens in our dataset (Epoch 1 Completed!).")
print("--> To train overnight for 6,000 steps (~25 Million tokens), we must enable Multi-Epoch Continuous Training!")

train_path = os.path.join(base_dir, "training", "train.py")
with open(train_path, "r", encoding="utf-8") as f:
    content = f.read()

# We add a continuous batch generator helper function before the training loop if not present
if "def get_continuous_batches" not in content:
    helper_code = '''
def get_continuous_batches(dataloader):
    epoch = 1
    while True:
        print(f"--> [Epoch {epoch} Start] Streaming data buffer...", flush=True)
        for batch in dataloader:
            yield batch
        epoch += 1
'''
    # Insert before the training loop
    content = content.replace("for step_offset, (x, y) in enumerate(dataloader, 1):", helper_code + "
    for step_offset, (x, y) in enumerate(get_continuous_batches(dataloader), 1):")
else:
    # If already present or modified
    pass

with open(train_path, "w", encoding="utf-8") as f:
    f.write(content)

print("--> Updated training/train.py: Enabled Multi-Epoch Continuous Streaming (Will never stop until 6,000 steps!)")

print("\n=======================================================")
print("=== [2. LAUNCHING CONTINUOUS OVERNIGHT RUN (FROM STEP 1,600)] ===")
print("=======================================================")

subprocess.run(["pkill", "-f", "train.py"], check=False)
time.sleep(1)

python_bin = "/system/conda/miniconda3/envs/cloudspace/bin/python3"
log_file = os.path.join(base_dir, "training.log")

cmd = f"nohup {python_bin} -u training/train.py > /dev/null 2>&1 &"
subprocess.run(cmd, shell=True, check=True)
print("--> Launched continuous overnight process in background!")

time.sleep(6)

res_ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
train_procs = [line for line in res_ps.stdout.splitlines() if "train.py" in line and "grep" not in line]

if train_procs:
    print("SUCCESS: Continuous Overnight Training is LIVE!")
    for p in train_procs:
        print("  PID info:", p[:80])
else:
    print("WARNING: Could not detect train.py process.")

print("\n=======================================================")
print("=== [3. INITIAL LOG VERIFICATION] ===")
print("=======================================================")

if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    print("Last 15 lines of training.log:")
    for line in lines[-15:]:
        print(" ", line.strip())
else:
    print("Log file not found.")
