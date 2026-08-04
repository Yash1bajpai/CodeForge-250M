import os
import shutil

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
ckpt_dir = os.path.join(base_dir, "checkpoints", "CodeForge-250M")

print("=================================================================")
print("=== [PRUNING ANCIENT CHECKPOINTS TO PREVENT DISK OVERFLOW] ===")
print("=================================================================")

keep_milestones = [55000, 61000, 67000, 73000]

if not os.path.exists(ckpt_dir):
    print(f"Error: {ckpt_dir} not found.")
    exit(1)

files = os.listdir(ckpt_dir)
deleted_count = 0
freed_bytes = 0

for fname in sorted(files):
    fpath = os.path.join(ckpt_dir, fname)
    if not os.path.isfile(fpath):
        continue
    
    # Keep special files
    if fname in ["latest_checkpoint.pt", "checkpoint_chunk1.pt"]:
        print(f"  [KEEP - Special] {fname}")
        continue
    
    if fname.startswith("checkpoint_step_") and fname.endswith(".pt"):
        try:
            step_str = fname.replace("checkpoint_step_", "").replace(".pt", "")
            step_val = int(step_str)
            
            # Keep if it's a major milestone OR if it's a very recent step >= 67000 (from Chunk 14/15)
            if step_val in keep_milestones or step_val >= 67000:
                print(f"  [KEEP - Milestone/Recent] {fname} (Step {step_val})")
            else:
                size_bytes = os.path.getsize(fpath)
                os.remove(fpath)
                deleted_count += 1
                freed_bytes += size_bytes
                print(f"  [DELETED] {fname} ({size_bytes / (1024**3):.2f} GB)")
        except ValueError:
            print(f"  [SKIP - Unparsed] {fname}")

print("\n=================================================================")
print(f"--> PRUNING SUMMARY: Deleted {deleted_count} ancient checkpoint files!")
print(f"--> TOTAL FREED SPACE: {freed_bytes / (1024**3):.2f} GB (~{freed_bytes / (1024**3):.0f} GB freed!)")
print("=================================================================")

print("\n--> Current Cloud SSD Disk Usage:")
os.system(f"df -h {base_dir}")
