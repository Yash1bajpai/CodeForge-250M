#!/bin/bash
# Setup environment for CodeForge-250M on Lightning AI / Linux
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== [CodeForge-250M] Starting Environment Setup ==="

# 1. Update pip and install requirements
echo "--> Installing Python dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r "$PROJECT_ROOT/requirements.txt"

# 2. Check GPU availability and CUDA version
echo "--> Checking CUDA / GPU Status..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()} | Device Count: {torch.cuda.device_count()} | Current Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# 3. Setup HuggingFace CLI Login using environment variable or token
if [ -z "$HF_TOKEN" ]; then
    echo "--> [Warning] HF_TOKEN environment variable not set. If pushing checkpoints to HF Hub, please set export HF_TOKEN='your_token'."
else
    echo "--> Logging into Hugging Face Hub..."
    python3 -c "from huggingface_hub import login; import os; login(token=os.environ.get('HF_TOKEN'), add_to_git_credential=True); print('HF Hub Login Successful!')"
fi

echo "=== [CodeForge-250M] Environment Setup Complete! ==="
