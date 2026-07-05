#!/bin/bash
# Setup environment for CodeForge-250M on Lightning AI / Linux
set -e

echo "=== [CodeForge-250M] Starting Environment Setup ==="

# 1. Update pip and install requirements
echo "--> Installing Python dependencies..."
pip install --upgrade pip
pip install -r /teamspace/studios/this_studio/CodeForge-250M/requirements.txt

3 2. Check GPU availability and CUDA version
echo "--> Checking CUDA / GPU Status..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()} | Device Count: {torch.cuda.device_count()} | Current Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# 3. Setup HuggingFace CLI Login using environment variable or token
if [ -z "$HF_TOKEN" ]; then
    echo "--> HF_TOKEN not set in environment. Setting default token for Yash1bajpai..."
    export HF_TOKEN=""$HF_TOKEN""
fi
echo "--> Logging into Hugging Face Hub..."
python3 -c "from huggingface_hub import login; import os; login(token=os.environ.get('HF_TOKEN', '"$HF_TOKEN"'), add_to_git_credential=True); print('HF Hub Login Successful!')"

echo "=== [CodeForge-250M] Environment Setup Complete! ==="
