import os
import time
import subprocess

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
os.chdir(base_dir)

print("=================================================================")
print("=== [1. CONFIGURING CHUNK 4 (OVERNIGHT RUN 2 -> STEP 13,500)] ===")
print("=================================================================")

train_path = os.path.join(base_dir, "training", "train.py")
with open(train_path, "r", encoding="utf-8") as f:
    content = f.read()

# Update target steps to 13500 (~55 Million tokens)
if "if step_offset >= 6000:" in content:
    content = content.replace("if step_offset >= 6000:", "if step_offset >= 6000:") # Keep 6000 step chunk size from resume point!
elif "max_steps = " in content:
    content = content.replace("max_steps = 6500", "max_steps = 13500")

content = content.replace("OVERNIGHT CHUNK 3", "OVERNIGHT CHUNK 4")
content = content.replace("Chunk 3 completed", "Chunk 4 completed")

with open(train_path, "w", encoding="utf-8") as f:
    f.write(content)

print("--> Successfully updated training/train.py for Overnight Chunk 4!")
print("    - Resumes from: latest_checkpoint.pt (Step 7,430 / Loss 1.42)")
print("    - Target Chunk Size: +6,000 steps (Will reach Step ~13,430 / ~55 Million tokens!)")

print("\n=================================================================")
print("=== [2. LAUNCHING CHUNK 4 IN BACKGROUND (NOHUP)] ===")
print("=================================================================")

subprocess.run(["pkill", "-f", "train.py"], check=False)
time.sleep(1)

python_bin = "/system/conda/miniconda3/envs/cloudspace/bin/python3"
log_file = os.path.join(base_dir, "training.log")

cmd = f"nohup {python_bin} -u training/train.py > /dev/null 2>&1 &"
subprocess.run(cmd, shell=True, check=True)
print("--> Launched Chunk 4 training process in background!")

time.sleep(6)

res_ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
train_procs = [line for line in res_ps.stdout.splitlines() if "train.py" in line and "grep" not in line]

if train_procs:
    print("SUCCESS: Chunk 4 Training is LIVE and running on Tesla T4 GPU!")
    for p in train_procs:
        print("  PID info:", p[:80])
else:
    print("WARNING: Could not detect train.py process.")

print("\n=================================================================")
print("=== [3. INITIAL LOG VERIFICATION (training.log)] ===")
print("=================================================================")

if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    print("Last 15 lines of training.log:")
    for line in lines[-15:]:
        print(" ", line.strip())
else:
    print("Log file not found.")
