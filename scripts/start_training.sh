#!/bin/bash
# Master training launch script for CodeForge-250M
export PATH="/system/conda/miniconda3/bin:$PATH"
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== [CodeForge-250M] Launching Distributed Training Pipeline ==="
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# Launch training via Accelerate / DeepSpeed ZeRO-2
echo "--> Executing training/train.py with config configs/config_250M.yaml..."
python3 "$PROJECT_ROOT/training/train.py" "$@"

echo "=== [CodeForge-250M] Training Execution Finished ==="
