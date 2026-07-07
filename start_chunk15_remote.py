import os
import time
import subprocess

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
os.chdir(base_dir)

print("=================================================================")
print("=== [1. VERIFYING CHUNK 14 (~295M TOKENS!) & CLEANING DISK] ===")
print("=================================================================")

log_file = os.path.join(base_dir, "training.log")
if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    print("Last 15 lines of training.log:")
    for line in lines[-15:]:
        print(" ", line.strip())

print("\n--> Cleaning up intermediate checkpoint files (retaining only milestones)...")
ckpt_dir = os.path.join(base_dir, "checkpoints", "CodeForge-250M")
keep_milestones = ["latest_checkpoint.pt", "checkpoint_chunk1.pt", "checkpoint_step_55000.pt", "checkpoint_step_61000.pt", "checkpoint_step_67000.pt", "checkpoint_step_72000.pt"]
if os.path.exists(ckpt_dir):
    for fname in os.listdir(ckpt_dir):
        if fname.startswith("checkpoint_step_") and fname.endswith(".pt") and fname not in keep_milestones:
            fpath = os.path.join(ckpt_dir, fname)
            try:
                os.remove(fpath)
                print(f"  Deleted intermediate bloat: {fname}")
            except Exception as e:
                print(f"  Error deleting {fname}: {e}")

print("\n=================================================================")
print("=== [2. RELEASING CHUNK 14 (~300M MARK!) TO HUGGING FACE HUB] ===")
print("=================================================================")

python_bin = "/system/conda/miniconda3/envs/cloudspace/bin/python3"
subprocess.run([python_bin, "auto_release_chunk3.py"], check=False)

print("\n=================================================================")
print("=== [3. CONFIGURING STAGE 15 / CHUNK 15 (TARGET -> STEP 78,000)] ===")
print("=== [TOWARDS 320 MILLION TOKENS!] ===")
print("=================================================================")

train_path = os.path.join(base_dir, "training", "train.py")
with open(train_path, "r", encoding="utf-8") as f:
    content = f.read()

# Update target steps to 78000 (~320 Million tokens!)
if "max_steps = " in content:
    content = content.replace("max_steps = 73000", "max_steps = 78000")
    content = content.replace("max_steps = 72000", "max_steps = 78000")
    content = content.replace("max_steps = 67000", "max_steps = 78000")

content = content.replace("STAGE 14 / CHUNK 14", "STAGE 15 / CHUNK 15")
content = content.replace("Chunk 14 completed", "Chunk 15 completed")

# Ensure checkpoint saving is set to 1000 steps
content = content.replace("if step % 100 == 0:", "if step % 1000 == 0:")

with open(train_path, "w", encoding="utf-8") as f:
    f.write(content)

print("--> Successfully updated training/train.py for Stage 15 / Chunk 15!")
print("    - Resumes from: latest_checkpoint.pt (~Step 72,000 / ~295 Million Tokens!)")
print("    - Target Chunk Size: +6,000 steps (Will reach Step 78,000 / ~320 MILLION TOKENS!)")
print("--> READY FOR LAUNCH ON T4 OR FOR 1-CLICK SWITCH TO L40S / A100 GPU!")
