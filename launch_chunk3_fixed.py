import os
import time
import subprocess

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
os.chdir(base_dir)

print("=======================================================")
print("=== [1. FIXING train.py FOR OVERNIGHT CHUNK 3 (6,000 STEPS)] ===")
print("=======================================================")

train_path = os.path.join(base_dir, "training", "train.py")
with open(train_path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1: Load latest_checkpoint.pt instead of checkpoint_chunk1.pt
content = content.replace('chunk1_path = os.path.join(ckpt_dir, "checkpoint_chunk1.pt")', 'latest_path = os.path.join(ckpt_dir, "latest_checkpoint.pt")')
content = content.replace('if os.path.exists(chunk1_path):', 'if os.path.exists(latest_path):')
content = content.replace('print(f"--> [Resume] Resuming weights from Chunk 1 Checkpoint: {chunk1_path}...")', 'print(f"--> [Resume] Resuming weights from Latest Checkpoint: {latest_path}...")')
content = content.replace('ckpt = torch.load(chunk1_path, map_location=device)', 'ckpt = torch.load(latest_path, map_location=device)')

# Fix 2: Increase DataLoader sample limit from 5000 to 50000
content = content.replace("if len(self.samples) >= 5000:", "if len(self.samples) >= 50000:")
content = content.replace("if len(self.samples) >= 5000:", "if len(self.samples) >= 50000:")

# Fix 3: Change loop break from 500 steps to 6000 steps
content = content.replace("if step_offset >= 500:  # Train 500 steps (2 Million tokens) for Chunk 2", "if step_offset >= 6000:  # Train 6000 steps (~25 Million tokens) for Overnight Chunk 3")
content = content.replace("if step_offset >= 500:", "if step_offset >= 6000:")

# Fix 4: Update print titles
content = content.replace("=== [LAUNCHING CODEFORGE-250M TRAINING CHUNK 2] ===", "=== [LAUNCHING CODEFORGE-250M OVERNIGHT CHUNK 3] ===")
content = content.replace("--> [CHUNK 2 COMPLETED]", "--> [OVERNIGHT CHUNK 3 COMPLETED]")
content = content.replace("Chunk 2 completed", "Overnight Chunk 3 completed")

with open(train_path, "w", encoding="utf-8") as f:
    f.write(content)

print("--> Successfully updated training/train.py:")
print("    - Resumes from: latest_checkpoint.pt (Step 500)")
print("    - Target step count: +6,000 steps (~25 Million tokens overnight!)")

print("\n=======================================================")
print("=== [2. LAUNCHING CHUNK 3 IN BACKGROUND] ===")
print("=======================================================")

subprocess.run(["pkill", "-f", "train.py"], check=False)
time.sleep(1)

python_bin = "/system/conda/miniconda3/envs/cloudspace/bin/python3"
log_file = os.path.join(base_dir, "training.log")

# Clear old log or append
cmd = f"nohup {python_bin} -u training/train.py > /dev/null 2>&1 &"
subprocess.run(cmd, shell=True, check=True)
print("--> Launched Chunk 3 training process in background!")

time.sleep(6)

res_ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
train_procs = [line for line in res_ps.stdout.splitlines() if "train.py" in line and "grep" not in line]

if train_procs:
    print("SUCCESS: Chunk 3 training is LIVE and running!")
    for p in train_procs:
        print("  PID info:", p[:80])
else:
    print("WARNING: Could not detect train.py process.")

print("\n=======================================================")
print("=== [3. LIVE LOG VERIFICATION (training.log)] ===")
print("=======================================================")

if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    print("Last 15 lines of training.log:")
    for line in lines[-15:]:
        print(" ", line.strip())
else:
    print("Log file not found.")
