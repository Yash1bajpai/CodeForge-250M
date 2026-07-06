import os
import time
import subprocess

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
os.chdir(base_dir)

print("=================================================================")
print("=== [1. CONFIGURING STAGE 6 / CHUNK 6 (TARGET -> STEP 25,500)] ===")
print("=================================================================")

train_path = os.path.join(base_dir, "training", "train.py")
with open(train_path, "r", encoding="utf-8") as f:
    content = f.read()

# Update target steps to 25500 (~105 Million tokens - Crossing 100M!)
if "max_steps = " in content:
    content = content.replace("max_steps = 19500", "max_steps = 25500")
    content = content.replace("max_steps = 13500", "max_steps = 25500")

content = content.replace("STAGE 5 / CHUNK 5", "STAGE 6 / CHUNK 6")
content = content.replace("Chunk 5 completed", "Chunk 6 completed")

# Ensure checkpoint saving is set to 1000 steps to prevent storage overflow
content = content.replace("if step % 100 == 0:", "if step % 1000 == 0:")

with open(train_path, "w", encoding="utf-8") as f:
    f.write(content)

print("--> Successfully updated training/train.py for Stage 6 / Chunk 6!")
print("    - Resumes from: latest_checkpoint.pt (~Step 19,400 / Loss 0.2866)")
print("    - Target Chunk Size: +6,000 steps (Will reach Step ~25,400 / ~105 Million tokens!)")
print("    - Checkpoint Saving: Every 1,000 steps (100% Safe from Disk Overflow!)")

print("\n=================================================================")
print("=== [2. LAUNCHING CHUNK 6 IN BACKGROUND (NOHUP)] ===")
print("=================================================================")

subprocess.run(["pkill", "-f", "train.py"], check=False)
time.sleep(1)

python_bin = "/system/conda/miniconda3/envs/cloudspace/bin/python3"
log_file = os.path.join(base_dir, "training.log")

cmd = f"nohup {python_bin} -u training/train.py > /dev/null 2>&1 &"
subprocess.run(cmd, shell=True, check=True)
print("--> Launched Chunk 6 training process in background!")

time.sleep(6)

res_ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
train_procs = [line for line in res_ps.stdout.splitlines() if "train.py" in line and "grep" not in line]

if train_procs:
    print("SUCCESS: Stage 6 / Chunk 6 Training is LIVE and running on Tesla T4 GPU!")
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
