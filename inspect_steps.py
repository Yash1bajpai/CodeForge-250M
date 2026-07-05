import os

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
train_path = os.path.join(base_dir, "training", "train.py")

with open(train_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print("=== [INSPECTING training/train.py FOR STEP LIMITS] ===")
for idx, line in enumerate(lines):
    if any(k in line.lower() for k in ["step", "chunk", "range", "while", "break", "550", "500", "1000"]):
        print(f"Line {idx+1}: {line.rstrip()}")
