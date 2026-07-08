import os
from huggingface_hub import HfApi

def main():
    print("--> [HF Hub Push] Initiating upload of Step 96,200 checkpoint...")
    ckpt_path = "checkpoints/CodeForge-250M/checkpoint_step_96200.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: Could not find {ckpt_path}")
        return

    api = HfApi()
    try:
        api.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo="checkpoints/checkpoint_step_96200.pt",
            repo_id="Yash1bajpai/CodeForge-250M",
            commit_message="Checkpoint step 96200 (~0.81B tokens - Stage 18 RTXP 6000 Sprint)"
        )
        print("--> [SUCCESS] Step 96,200 checkpoint successfully pushed to Hugging Face Hub!")
    except Exception as e:
        print(f"--> [Error] Push failed: {e}")

if __name__ == "__main__":
    main()
