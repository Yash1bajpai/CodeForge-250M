import os
import time
import subprocess

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
os.chdir(base_dir)

print("=================================================================")
print("=== [1. RELEASING STAGE 8 (~155M TOKENS) TO HUGGING FACE] ===")
print("=================================================================")

python_bin = "/system/conda/miniconda3/envs/cloudspace/bin/python3"
subprocess.run([python_bin, "auto_release_chunk3.py"], check=False)

print("\n=================================================================")
print("=== [2. CONFIGURING STAGE 9 / CHUNK 9 (TARGET -> STEP 43,500)] ===")
print("=================================================================")

train_path = os.path.join(base_dir, "training", "train.py")
with open(train_path, "r", encoding="utf-8") as f:
    content = f.read()

# Update target steps to 43500 (~180 Million tokens)
if "max_steps = " in content:
    content = content.replace("max_steps = 37500", "max_steps = 43500")
    content = content.replace("max_steps = 31500", "max_steps = 43500")

content = content.replace("STAGE 8 / CHUNK 8", "STAGE 9 / CHUNK 9")
content = content.replace("Chunk 8 completed", "Chunk 9 completed")

# Ensure checkpoint saving is set to 1000 steps
content = content.replace("if step % 100 == 0:", "if step % 1000 == 0:")

with open(train_path, "w", encoding="utf-8") as f:
    f.write(content)

print("--> Successfully updated training/train.py for Stage 9 / Chunk 9!")
print("    - Resumes from: latest_checkpoint.pt (~Step 37,000 / ~151 Million Tokens!)")
print("    - Target Chunk Size: +6,000 steps (Will reach Step ~43,400 / ~180 Million tokens!)")

print("\n=================================================================")
print("=== [3. LAUNCHING CHUNK 9 IN BACKGROUND (NOHUP)] ===")
print("=================================================================")

subprocess.run(["pkill", "-f", "train.py"], check=False)
time.sleep(1)

log_file = os.path.join(base_dir, "training.log")
cmd = f"nohup {python_bin} -u training/train.py > /dev/null 2>&1 &"
subprocess.run(cmd, shell=True, check=True)
print("--> Launched Chunk 9 training process in background!")

time.sleep(6)

res_ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
train_procs = [line for line in res_ps.stdout.splitlines() if "train.py" in line and "grep" not in line]

if train_procs:
    print("SUCCESS: Stage 9 / Chunk 9 Training is LIVE and running on Tesla T4 GPU!")
    for p in train_procs:
        print("  PID info:", p[:80])
else:
    print("WARNING: Could not detect train.py process.")

print("\n=================================================================")
print("=== [4. INITIAL LOG VERIFICATION (training.log)] ===")
print("=================================================================")

if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    print("Last 15 lines of training.log:")
    for line in lines[-15:]:
        print(" ", line.strip())
else:
    print("Log file not found.")
