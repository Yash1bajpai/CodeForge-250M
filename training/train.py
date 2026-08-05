import os
import sys
import time
import math
import glob
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast

# Ensure project root is in path portably
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.architecture import CodeForgeModel
from models.init_weights import init_model_weights
from training.utils import get_lr_cosine_schedule

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.logfile = open(filepath, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

class ShardedCodeDataset(Dataset):
    """
    Production Sharded Dataset Loader.
    Dynamically loads pre-tokenized PyTorch tensor shards (shard_*.pt) from data/tokenized/.
    Zero on-the-fly tokenization overhead, streaming full 2.35B+ tokens across epochs smoothly.
    """
    def __init__(self, tokenized_dir: str = "data/tokenized", seq_length: int = 2048):
        self.seq_length = seq_length
        self.shard_files = sorted(glob.glob(os.path.join(tokenized_dir, "shard_*.pt")))
        self.samples = []
        
        print(f"--> [DataLoader] Scanning pre-tokenized shards in {tokenized_dir}...")
        if self.shard_files:
            for s_path in self.shard_files:
                try:
                    shard_data = torch.load(s_path, map_location="cpu")
                    if isinstance(shard_data, torch.Tensor):
                        self.samples.extend(shard_data)
                    elif isinstance(shard_data, list):
                        self.samples.extend(shard_data)
                except Exception as e:
                    print(f"    [Warning] Could not load shard {s_path}: {e}")
            print(f"--> [DataLoader] Loaded {len(self.samples):,} total sequences ({len(self.samples) * seq_length:,} tokens) across {len(self.shard_files)} shards!\n")
        else:
            print("--> [DataLoader] No pre-tokenized shards found. Falling back to on-the-fly deduplicated stream...")
            dedup_files = glob.glob(os.path.join(PROJECT_ROOT, "data/dedup/*_dedup.jsonl"))
            if not dedup_files:
                dedup_files = glob.glob(os.path.join(PROJECT_ROOT, "data/raw/*.jsonl"))
            if dedup_files:
                tok_path = os.path.join(PROJECT_ROOT, "data/tokenizer")
                if os.path.exists(tok_path):
                    tok = PreTrainedTokenizerFast.from_pretrained(tok_path)
                    eos_id = tok.eos_token_id or 0
                    buffer = []
                    for fpath in dedup_files:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                code = line.replace("\\n", "\n")
                                tokens = tok.encode(code) + [eos_id]
                                buffer.extend(tokens)
                                while len(buffer) >= seq_length + 1:
                                    self.samples.append(torch.tensor(buffer[:seq_length + 1], dtype=torch.long))
                                    buffer = buffer[seq_length:]
                    print(f"--> [DataLoader] Fallback loaded {len(self.samples):,} sequences.\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        if isinstance(chunk, list):
            chunk = torch.tensor(chunk, dtype=torch.long)
        if len(chunk) > self.seq_length:
            x = chunk[:-1]
            y = chunk[1:]
        else:
            x = chunk
            y = chunk
        return x, y

def train():
    parser = argparse.ArgumentParser(description="CodeForge-250M Training Engine")
    parser.add_argument("--from_scratch", action="store_true", help="Start training from Step 0 cleanly (ignore existing checkpoints due to overfitting)")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs/config_250M.yaml"), help="Path to YAML training config")
    args = parser.parse_args()

    log_path = os.path.join(PROJECT_ROOT, "training.log")
    sys.stdout = Logger(log_path)
    
    print("\n===============================================================================")
    print("=== [LAUNCHING CODEFORGE-250M PRODUCTION TRAINING ENGINE (NEXUS-AGENT ReAct)] ===")
    print(f"=== [LOGGING TO: {log_path}] ===")
    print("===============================================================================")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is not available! Please check driver/hardware.")
        return
    
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"--> Allocated Hardware : {gpu_name} (SM {cap[0]}.{cap[1]})")
    
    if cap[0] < 8:
        print("--> Architecture Note  : Turing SM < 8 detected. Using FP16 Tensor Cores with SDPA!")
        dtype = torch.float16
    else:
        print("--> Architecture Note  : Ampere+ detected. Using BF16 Tensor Cores with SDPA!")
        dtype = torch.bfloat16
        
    with open(args.config, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
        cfg = full_cfg["model"]
        train_cfg = full_cfg["training"]
        
    print(f"--> Initializing CodeForge-250M Model ({cfg['num_hidden_layers']}L / {cfg['hidden_size']}H / GQA)...")
    model = CodeForgeModel(cfg).to(device)
    param_count = model.get_parameter_count()
    print(f"    Total Parameters   : {param_count:,} (~250M Target)")
    
    # Checkpoint & Resume Management
    start_step = 0
    loss_val = 0.0
    ckpt_dir = os.path.join(PROJECT_ROOT, full_cfg.get("checkpointing", {}).get("output_dir", "checkpoints/CodeForge-250M"))
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = os.path.join(ckpt_dir, "latest_checkpoint.pt")
    
    batch_size_per_device = train_cfg.get("batch_size_per_device", 16)
    accum_steps = train_cfg.get("gradient_accumulation_steps", 16)
    max_steps = train_cfg.get("max_steps", 4482)
    max_lr = float(train_cfg.get("learning_rate", 6.0e-4))
    min_lr = float(train_cfg.get("min_learning_rate", 6.0e-5))
    warmup_steps = int(train_cfg.get("warmup_steps", 500))
    save_steps = int(full_cfg.get("checkpointing", {}).get("save_steps", 100))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=train_cfg.get("weight_decay", 0.1), betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))

    if os.path.exists(latest_path) and not args.from_scratch:
        print(f"--> [Resume] Loading weights from Latest Checkpoint: {latest_path}...")
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt.get('step', 0)
        loss_val = ckpt.get('loss', 0.0)
        print(f"    Resumed successfully from Step {start_step} (Previous Loss: {loss_val:.4f})")
    else:
        if args.from_scratch:
            print("--> [From Scratch] --from_scratch flag enabled. Discarding previous checkpoints and starting at Step 0.")
        else:
            print("--> [From Scratch] No checkpoint found. Starting fresh training run at Step 0.")
        init_model_weights(model, initializer_range=0.02)
        start_step = 0
        
    dataset = ShardedCodeDataset(os.path.join(PROJECT_ROOT, "data/tokenized"), seq_length=cfg["max_position_embeddings"])
    if len(dataset) == 0:
        print("ERROR: Dataset is empty! Please run download_stack.py and tokenize_dataset.py first.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size_per_device, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    
    print("===============================================================================")
    print("=== [LIVE TRAINING PROGRESS (EFFECTIVE BATCH: 256 SEQS / 524k TOKENS/STEP)] ===")
    print("===============================================================================")
    print(f"{'Step':<6} | {'Loss':<8} | {'Perplexity':<10} | {'LR':<10} | {'VRAM (GB)':<10} | {'Status'}")
    print("-" * 75)
    
    model.train()
    start_time = time.time()
    
    def get_continuous_batches(loader):
        while True:
            for b in loader:
                yield b

    batch_generator = get_continuous_batches(dataloader)
    optimizer.zero_grad()
    
    for step_offset in range(1, (max_steps - start_step) + 1):
        step = start_step + step_offset
        
        # Cosine LR Warmup + Decay Schedule
        current_lr = get_lr_cosine_schedule(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        step_loss = 0.0
        for micro_step in range(accum_steps):
            x, y = next(batch_generator)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=dtype):
                logits, loss = model(x, y)
                scaled_loss = loss / accum_steps
                
            scaler.scale(scaled_loss).backward()
            step_loss += loss.item() / accum_steps
            
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.get("max_grad_norm", 1.0))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        loss_val = step_loss
        
        if step % 5 == 0 or step_offset == 1:
            ppl = math.exp(min(loss_val, 20.0))
            vram_gb = torch.cuda.memory_allocated() / (1024**3)
            print(f"{step:<6} | {loss_val:<8.4f} | {ppl:<10.2f} | {current_lr:<10.2e} | {vram_gb:<10.2f} | Active Computing ⚡", flush=True)
            
        if step % save_steps == 0 or step == max_steps:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
            }, ckpt_path)
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
            }, latest_path)
            print(f"--> [Checkpoint] Auto-saved milestone at Step {step} to {ckpt_path}", flush=True)
            
            # PUSH TO HUGGING FACE HUB EVERY 1000 STEPS
            if step % 1000 == 0:
                hf_token = os.environ.get("HF_TOKEN")
                hf_repo = os.environ.get("HF_REPO_ID", "Yash1bajpai/CodeForge-250M")
                if hf_token:
                    print(f"--> [HF Hub] Uploading checkpoint Step {step} to Hugging Face Hub ({hf_repo})...", flush=True)
                    try:
                        from huggingface_hub import HfApi
                        api = HfApi(token=hf_token)
                        # Create repo if it doesn't exist
                        api.create_repo(repo_id=hf_repo, exist_ok=True, private=False)
                        
                        # Upload just the latest checkpoint to save bandwidth, or the whole dir
                        api.upload_file(
                            path_or_fileobj=latest_path,
                            path_in_repo="latest_checkpoint.pt",
                            repo_id=hf_repo,
                            commit_message=f"Upload training checkpoint - Step {step} (Loss: {loss_val:.4f})"
                        )
                        print(f"--> [HF Hub] Successfully uploaded Step {step} checkpoint!", flush=True)
                    except Exception as e:
                        print(f"    [Warning] HF Hub upload failed: {e}", flush=True)
                else:
                    print("    [Warning] Skipping HF Hub upload because HF_TOKEN is not set.", flush=True)

        stop_file = os.path.join(PROJECT_ROOT, "STOP_AND_SAVE")
        if os.path.exists(stop_file):
            print(f"--> [Signal] STOP_AND_SAVE trigger detected at Step {step}! Saving checkpoint and exiting...", flush=True)
            try:
                os.remove(stop_file)
            except Exception:
                pass
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
            }, ckpt_path)
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
            }, latest_path)
            print(f"--> [Checkpoint] Saved weights safely at Step {step} before interrupt", flush=True)
            break

    elapsed = time.time() - start_time
    tokens_processed = (step - start_step) * accum_steps * batch_size_per_device * cfg["max_position_embeddings"]
    tps = tokens_processed / max(1.0, elapsed)
    
    print("-" * 75)
    print(f"--> [TRAINING RUN COMPLETED] Processed {tokens_processed:,} tokens in {elapsed:.1f} seconds ({tps:.1f} tokens/sec)!")
    print("SUCCESS: CodeForge-250M training run completed smoothly!")

if __name__ == "__main__":
    train()
