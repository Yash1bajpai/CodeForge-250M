import os
import subprocess
from huggingface_hub import HfApi, login

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
os.chdir(base_dir)

print("=======================================================")
print("=== [1. GITHUB REPOSITORY & GIT STATUS CHECK] ===")
print("=======================================================")

# Ensure .gitignore prevents pushing heavy data/checkpoints to GitHub
gitignore_path = os.path.join(base_dir, ".gitignore")
with open(gitignore_path, "w", encoding="utf-8") as f:
    f.write("data/raw/\ndata/filtered/\ndata/dedup/\ndata/tokenized/\ncheckpoints/\n*.log\n*.pt\n__pycache__/\n")
print("--> Verified .gitignore (excluding 17GB checkpoints & raw data from Git)")

# Check git status and remote
res_status = subprocess.run(["git", "status"], capture_output=True, text=True)
res_remote = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
print("GIT STATUS:\n", res_status.stdout if res_status.returncode == 0 else res_status.stderr)
print("GIT REMOTE:\n", res_remote.stdout if res_remote.returncode == 0 else "No remote configured yet.")

print("\n=======================================================")
print("=== [2. UPLOADING MODEL & TOKENIZER TO HUGGING FACE HUB] ===")
print("=======================================================")
hf_token = os.environ.get("HF_TOKEN", "")
repo_id = "Yash1bajpai/CodeForge-250M"

try:
    print(f"--> Logging into Hugging Face Hub...")
    login(token=hf_token, add_to_git_credential=False)
    api = HfApi()
    
    print(f"--> Creating/Verifying repository {repo_id} on Hugging Face...")
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    
    # Upload custom FIM tokenizer
    print(f"--> Uploading custom 32k FIM Tokenizer to {repo_id}...")
    api.upload_folder(
        folder_path="data/tokenizer",
        path_in_repo="tokenizer",
        repo_id=repo_id,
        commit_message="Upload custom 32k FIM Tokenizer with distinct UNK/PAD"
    )
    
    # Upload config files
    print(f"--> Uploading Architecture Configs (250M, 500M, 1B) to {repo_id}...")
    api.upload_folder(
        folder_path="configs",
        path_in_repo="configs",
        repo_id=repo_id,
        commit_message="Upload exact parameter scaling architecture configs"
    )
    
    # Upload latest checkpoint
    ckpt_path = "checkpoints/CodeForge-250M/latest_checkpoint.pt"
    if os.path.exists(ckpt_path):
        print(f"--> Uploading latest Chunk 2 trained checkpoint (2.8 GB) to {repo_id}...")
        api.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo="latest_checkpoint.pt",
            repo_id=repo_id,
            commit_message="Upload Chunk 2 trained checkpoint (Loss: 2.90, Step 550)"
        )
        print(f"\nSUCCESS: Model checkpoint and tokenizer are LIVE on Hugging Face!")
        print(f"👉 Hugging Face URL: https://huggingface.co/{repo_id}")
    else:
        print(f"    [Warning] {ckpt_path} not found.")
except Exception as e:
    print(f"ERROR uploading to Hugging Face Hub: {e}")
