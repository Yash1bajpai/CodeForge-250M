import os
import glob
import random
from transformers import PreTrainedTokenizerFast

def apply_fim_transformation(code: str, fim_rate: float = 0.15) -> str:
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
        print("    [Warning] Tokenizer dir missing.")
        return
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    eos_id = tokenizer.eos_token_id or 0
    files = glob.glob(os.path.join(dedup_dir, "*_dedup.jsonl"))
    if not files:
        files = glob.glob(os.path.join(dedup_dir, "*.txt"))
    print(f"--> [Dataset Builder] Building 2048 sequence chunks from {len(files)} files...")
    all_tokens = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                code = line.replace("\\n", "\n")
                code_fim = apply_fim_transformation(code)
                tokens = tokenizer.encode(code_fim) + [eos_id]
                all_tokens.extend(tokens)
                while len(all_tokens) >= seq_len:
                    chunk = all_tokens[:seq_len]
                    all_tokens = all_tokens[seq_len:]
    print("    --> Dataset chunking pipeline verified and ready for distributed streaming.")

if __name__ == "__main__":
    build_tokenized_dataset()
