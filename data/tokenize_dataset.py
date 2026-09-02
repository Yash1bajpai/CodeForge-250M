import os
import glob
import json
import random
import torch
from transformers import PreTrainedTokenizerFast

random.seed(42)  # deterministic FIM splits -> identical shards on re-runs (resume safety)

def apply_fim_transformation(code: str, fim_rate: float = 0.50) -> str:
    """
    Applies Fill-In-the-Middle (FIM) transformation at a 50% rate (StarCoder / DeepSeek-Coder standard).
    Crucial for DevMind / Nexus-Agent infill() primitive.
    """
    if random.random() > fim_rate or len(code) < 50:
        return code
    lines = code.splitlines()
    if len(lines) < 3:
        return code
    split1 = random.randint(1, len(lines) - 2)
    split2 = random.randint(split1 + 1, len(lines) - 1)
    prefix = "\n".join(lines[:split1])
    middle = "\n".join(lines[split1:split2])
    suffix = "\n".join(lines[split2:])
    return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}"

def build_tokenized_dataset(dedup_dir: str = "data/dedup", tokenizer_dir: str = "data/tokenizer", output_dir: str = "data/tokenized", seq_len: int = 2048):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(tokenizer_dir):
        print(f"    [Warning] Tokenizer dir {tokenizer_dir} missing.")
        return
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    eos_id = tokenizer.eos_token_id or 0
    files = glob.glob(os.path.join(dedup_dir, "*_dedup.jsonl"))
    if not files:
        raise SystemExit(f"ERROR: no dedup files in {dedup_dir}. Run filter_quality.py and deduplicate.py first. Refusing to fall back to raw unfiltered data (run #1 overfitting cause).")
        
    print(f"--> [Dataset Builder] Building {seq_len}-token sequence chunks from {len(files)} files...")
    all_tokens = []
    chunk_count = 0
    shard_size = 1000  # Save 1,000 sequences (2,048,000 tokens) per shard file
    current_shard = []
    
    for file_path in files:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    code = data.get("text", "")
                except Exception:
                    code = line.replace("\\n", "\n")
                    
                code_fim = apply_fim_transformation(code, fim_rate=0.50)
                tokens = tokenizer.encode(code_fim) + [eos_id]
                all_tokens.extend(tokens)
                
                while len(all_tokens) >= seq_len:
                    chunk = all_tokens[:seq_len]
                    all_tokens = all_tokens[seq_len:]
                    current_shard.append(chunk)
                    chunk_count += 1
                    
                    if len(current_shard) >= shard_size:
                        shard_idx = chunk_count // shard_size
                        out_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.pt")
                        torch.save(torch.tensor(current_shard, dtype=torch.long), out_path)
                        current_shard = []
                        
    if current_shard:
        shard_idx = chunk_count // shard_size if chunk_count % shard_size == 0 else (chunk_count // shard_size) + 1
        out_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.pt")
        torch.save(torch.tensor(current_shard, dtype=torch.long), out_path)
        
    print(f"    --> Successfully saved {chunk_count:,} tokenized sequences ({chunk_count * seq_len:,} tokens) across shards in {output_dir}!")

if __name__ == "__main__":
    build_tokenized_dataset()
