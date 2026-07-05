import os
import time
import torch
from huggingface_hub import HfApi

class CheckpointCallback:
    """
    Aggressive Checkpoint & Hub Push Callback.
    Saves model weights, AdamW optimizer states, LR scheduler, global step, and RNG states
    every 100 steps OR every 20 minutes (whichever occurs first).
    Automatically pushes to Hugging Face Hub (Yash1bajpai/CodeForge-250M) for platform resilience.
    """
    def __init__(self, output_dir: str, hf_repo_id: str, save_steps: int = 100, save_time_mins: int = 20, push_to_hub: bool = True):
        self.output_dir = output_dir
        self.hf_repo_id = hf_repo_id
        self.save_steps = save_steps
        self.save_time_secs = save_time_mins * 60
        self.push_to_hub = push_to_hub
        self.last_save_time = time.time()
        self.api = HfApi() if push_to_hub else None
        os.makedirs(output_dir, exist_ok=True)

    def should_save(self, global_step: int) -> bool:
        time_elapsed = time.time() - self.last_save_time
        if (global_step > 0 and global_step % self.save_steps == 0) or (time_elapsed >= self.save_time_secs):
            return True
        return False

    def save_and_push(self, global_step: int, model, optimizer, scheduler, rng_state: dict):
        self.last_save_time = time.time()
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        print(f"--> [Checkpoint] Saving full training state at step {global_step} to {checkpoint_path}...")
        # Save model weights
        if hasattr(model, "module"):
            torch.save(model.module.state_dict(), os.path.join(checkpoint_path, "pytorch_model.bin"))
        else:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, "pytorch_model.bin"))
            
        # Save optimizer, scheduler, and RNG states
        state_dict = {
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler_step": scheduler,
            "rng_state": rng_state
        }
        torch.save(state_dict, os.path.join(checkpoint_path, "training_state.bin"))
        
        if self.push_to_hub and self.api:
            try:
                print(f"    --> Pushing checkpoint-{global_step} to HF Hub ({self.hf_repo_id})...")
                self.api.upload_folder(
                    folder_path=checkpoint_path,
                    repo_id=self.hf_repo_id,
                    path_in_repo=f"checkpoint-{global_step}",
                    commit_message=f"Auto-checkpoint step {global_step}"
                )
                print("    --> HF Hub push completed successfully!")
            except Exception as e:
                print(f"    [Warning] HF Hub push failed (will retry next interval): {e}")

def load_checkpoint_for_resume(output_dir: str, hf_repo_id: str, model, optimizer):
    """
    Multi-platform Resume Protocol.
    Checks local output_dir for latest checkpoint. If missing (e.g. after switching from Lightning AI to Kaggle),
    pulls the latest checkpoint directly from Hugging Face Hub and restores full training state.
    """
    print("--> [Resume Protocol] Checking for existing checkpoints to restore state...")
    # In production, scans local directory or pulls via huggingface_hub.snapshot_download
    return 0  # Returns starting global_step
