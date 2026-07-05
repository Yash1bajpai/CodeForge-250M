#!/bin/bash
# Master training launch script for CodeForge-250M
set -e

echo "=== [CodeForge-250M] Launching Distributed Training Pipeline ==="
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# Launch training via Accelerate / DeepSpeed ZeRO-2
echo "--> Executing training/train.py with config configs/config_250M.yaml..."
python3 /teamspace/studios/this_studio/CodeForge-250M/training/train.py

echo "=== [CodeForge-250M] Training Execution Finished ==="
