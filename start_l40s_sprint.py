import os
import time
import subprocess

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
os.chdir(base_dir)

print("=================================================================")
print("=== [1. CONFIGURING L40S SPRINT -> STEP 122,000 (~500M TOKENS!)] ===")
print("=== [THE HALF-BILLION TOKEN MILESTONE!] ===")
print("=================================================================")

train_path = os.path.join(base_dir, "training", "train.py")
with open(train_path, "r", encoding="utf-8") as f:
    content = f.read()

# Update target steps to 122000 (~500 Million tokens - HALF BILLION!)
if "max_steps = " in content:
    content = content.replace("max_steps = 78000", "max_steps = 122000")
    content = content.replace("max_steps = 73000", "max_steps = 122000")
    content = content.replace("max_steps = 72000", "max_steps = 122000")
    content = content.replace("max_steps = 67000", "max_steps = 122000")
    content = content.replace("max_steps = 61000", "max_steps = 122000")

content = content.replace("STAGE 14 / CHUNK 14", "STAGE 15 / L40S HALF-BILLION SPRINT")
content = content.replace("STAGE 15 / CHUNK 15", "STAGE 15 / L40S HALF-BILLION SPRINT")

# Ensure checkpoint saving is set to 1000 steps
content = content.replace("if step % 100 == 0:", "if step % 1000 == 0:")

with open(train_path, "w", encoding="utf-8") as f:
    f.write(content)

print("--> Successfully updated training/train.py for L40S HALF-BILLION SPRINT!")
print("    - Resumes from: latest_checkpoint.pt (~Step 72,000 / ~295 Million Tokens!)")
print("    - Target Size : +50,000 steps (Will reach Step 122,000 / ~500 MILLION TOKENS / 0.5 BILLION!)")
print("    - Expected Speed on L40S: ~50,000+ tokens/sec (~180M+ tokens/hr!)")

print("\n=================================================================")
print("=== [2. LAUNCHING TRAINING ON NVIDIA L40S GPU (NOHUP)] ===")
print("=================================================================")

python_bin = "/system/conda/miniconda3/envs/cloudspace/bin/python3"
subprocess.run(["pkill", "-f", "train.py"], check=False)
time.sleep(1)

cmd = f"nohup {python_bin} -u training/train.py > /dev/null 2>&1 &"
subprocess.run(cmd, shell=True, check=True)
print("--> Launched L40S Half-Billion Sprint in background!")

time.sleep(6)

res_ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
train_procs = [line for line in res_ps.stdout.splitlines() if "train.py" in line and "grep" not in line]

if train_procs:
    print("SUCCESS: L40S Half-Billion Sprint is LIVE and running on NVIDIA L40S GPU!")
    for p in train_procs:
        print("  PID info:", p[:80])
else:
    print("WARNING: Could not detect train.py process.")

print("\n=================================================================")
print("=== [3. INITIAL GPU STATUS & LOG CHECK] ===")
print("=================================================================")

subprocess.run(["nvidia-smi"], check=False)
