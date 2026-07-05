import os
import time
import subprocess

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
os.chdir(base_dir)

print("=======================================================")
print("=== [1. ENABLING MULTI-EPOCH CONTINUOUS LOOP] ===")
print("=======================================================")

train_path = os.path.join(base_dir, "training", "train.py")
with open(train_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace the loop header with a continuous generator wrapper
old_loop = "for step_offset, (x, y) in enumerate(dataloader, 1):"

new_loop = """def get_continuous_batches(loader):
        while True:
            for b in loader:
                yield b

    for step_offset, (x, y) in enumerate(get_continuous_batches(dataloader), 1):"""

if "def get_continuous_batches" not in content and old_loop in content:
    content = content.replace(old_loop, new_loop)
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("--> Successfully patched training/train.py for Multi-Epoch Continuous Overnight Streaming!")
else:
    print("--> training/train.py is already patched or loop header not found.")

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

time.sleep(5)

res_ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
train_procs = [line for line in res_ps.stdout.splitlines() if "train.py" in line and "grep" not in line]

if train_procs:
    print("SUCCESS: Continuous Overnight Training is LIVE and running!")
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
