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
    lines = f.readlines()

new_lines = []
for line in lines:
    if "for step_offset, (x, y) in enumerate(dataloader, 1):" in line:
        if "def get_continuous_batches" not in "".join(new_lines):
            new_lines.append("    def get_continuous_batches(loader):
")
            new_lines.append("        while True:
")
            new_lines.append("            for b in loader:
")
            new_lines.append("                yield b

")
        new_lines.append("    for step_offset, (x, y) in enumerate(get_continuous_batches(dataloader), 1):
")
    else:
        new_lines.append(line)

with open(train_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("--> Successfully enabled Multi-Epoch Continuous Streaming in training/train.py!")

print("\n=======================================================")
print("=== [2. LAUNCHING OVERNIGHT CONTINUOUS RUN] ===")
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
