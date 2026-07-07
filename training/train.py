import os
import sys
sys.path.append("/teamspace/studios/this_studio/CodeForge-250M")
import time
import math
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from models.architecture import CodeForgeModel

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

class CodeDataset(Dataset):
    def __init__(self, tokenized_dir, seq_length=2048):
        self.seq_length = seq_length
        self.samples = []
        import glob
        dedup_files = glob.glob("data/dedup/*_dedup.jsonl")
        if not dedup_files:
            dedup_files = glob.glob("data/raw/*.jsonl")
            
        print(f"--> [DataLoader] Loading deduplicated shards into sequence buffers...")
        tok = PreTrainedTokenizerFast.from_pretrained("data/tokenizer")
        eos_id = tok.eos_token_id or 0
        
        buffer = []
        for fpath in dedup_files:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    code = line.replace("\\n", "\n")
                    tokens = tok.encode(code) + [eos_id]
                    buffer.extend(tokens)
                    while len(buffer) >= seq_length + 1:
                        self.samples.append(buffer[:seq_length + 1])
                        buffer = buffer[seq_length:]
                        if len(self.samples) >= 50000:  # Load 50000 sequence chunks for continuous training
                            break
            if len(self.samples) >= 50000:
                break
        print(f"--> [DataLoader] Prepared {len(self.samples):,} sequence chunks ({len(self.samples)*seq_length:,} tokens total)!\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train():
    log_path = "/teamspace/studios/this_studio/CodeForge-250M/training.log"
    sys.stdout = Logger(log_path)
    
    print("\n=======================================================")
    print(f"=== [LAUNCHING CODEFORGE-250M STAGE 17 / RTXP 6000 1-HOUR POWER SPRINT (BATCH SIZE 16)] ===")
    print(f"=== [LOGGING TO: {log_path}] ===")
    print("=======================================================")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is not available!")
        return
    
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"--> Allocated Hardware : {gpu_name} (SM {cap[0]}.{cap[1]})")
    
    if cap[0] < 8:
        print("--> Architecture Note  : Turing SM 7.5 detected. Using FP16 Tensor Cores with SDPA!")
        dtype = torch.float16
    else:
        print("--> Architecture Note  : Ampere+ detected. Using BF16 Tensor Cores with SDPA!")
        dtype = torch.bfloat16
        
    with open("configs/config_250M.yaml", "r") as f:
        cfg = yaml.safe_load(f)["model"]
        
    print(f"--> Initializing CodeForge-250M Model ({cfg['num_hidden_layers']}L / {cfg['hidden_size']}H / GQA)...")
    model = CodeForgeModel(cfg).to(device)
    param_count = model.get_parameter_count()
    print(f"    Total Parameters   : {param_count:,} (~246M Edge AI Target)")
    
    dataset = CodeDataset("data/tokenized", seq_length=cfg["max_position_embeddings"])
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    
    start_step = 0
    ckpt_dir = "checkpoints/CodeForge-250M"
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = os.path.join(ckpt_dir, "latest_checkpoint.pt")
    if os.path.exists(latest_path):
        print(f"--> [Resume] Resuming weights from Latest Checkpoint: {latest_path}...")
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt.get('step', 50)
        print(f"    Resumed successfully from Step {start_step} (Loss: {ckpt.get('loss', 'N/A')}!)")
        
    print("=======================================================")
    print("=== [LIVE TRAINING RESULTS & PROGRESS REPORT] ===")
    print("=======================================================")
    print(f"{'Step':<6} | {'Loss':<8} | {'Perplexity':<10} | {'LR':<10} | {'VRAM (GB)':<10} | {'Status'}")
    print("-" * 65)
    
    model.train()
    start_time = time.time()
    
    def get_continuous_batches(loader):
        while True:
            for b in loader:
                yield b

    for step_offset, (x, y) in enumerate(get_continuous_batches(dataloader), 1):
        step = start_step + step_offset
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=dtype):
            logits, loss = model(x, y)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if step % 5 == 0 or step_offset == 1:
            loss_val = loss.item()
            ppl = math.exp(min(loss_val, 20.0))
            vram_gb = torch.cuda.memory_allocated() / (1024**3)
            lr = optimizer.param_groups[0]['lr']
            print(f"{step:<6} | {loss_val:<8.4f} | {ppl:<10.2f} | {lr:<10.2e} | {vram_gb:<10.2f} | Active Computing ⚡", flush=True)
            
        if False:  # Zero disk I/O pause during 1-hour sprint! Saving only at session end.
            pass
            
        if step_offset >= 7200:  # Train 7,200 steps (~236 Million tokens) for 1-Hour RTXP 6000 Sprint!
            break
            
    elapsed = time.time() - start_time
    tokens_processed = step_offset * 16 * cfg["max_position_embeddings"]
    tps = tokens_processed / elapsed
    
    print("-" * 65)
    print(f"--> [STAGE 17 / RTXP 6000 1-HOUR POWER SPRINT COMPLETED] Processed {tokens_processed:,} tokens in {elapsed:.1f} seconds ({tps:.1f} tokens/sec)!")
    
    # Save checkpoint once at the very end of the session!
    final_step = start_step + step_offset
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{final_step}.pt")
    torch.save({
        'step': final_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_val,
    }, ckpt_path)
    torch.save({
        'step': final_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_val,
    }, os.path.join(ckpt_dir, "latest_checkpoint.pt"))
    print(f"--> [Checkpoint] Saved final session weights at Step {final_step} to {ckpt_path}", flush=True)
    print("SUCCESS: RTXP 6000 1-Hour Power Sprint completed and saved!")

if __name__ == "__main__":
    train()
