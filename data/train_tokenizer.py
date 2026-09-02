import os
import glob
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast

def dataset_iterator(files):
    for fpath in files:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                except Exception:
                    text = line.replace("\\n", "\n")
                if text:
                    yield text

def train_custom_bpe_tokenizer(data_dir: str = "data/dedup", output_dir: str = "data/tokenizer", vocab_size: int = 32000):
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(data_dir, "*_dedup.jsonl"))
    if not files:
        raise SystemExit(f"ERROR: no dedup files in {data_dir}. Run filter_quality.py and deduplicate.py first. Refusing to fall back to raw unfiltered data (run #1 overfitting cause).")

    print(f"--> [Tokenizer] Training 32k BPE Tokenizer with ReAct & FIM tokens on {len(files)} files...")
    
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    special_tokens = [
        "<|endoftext|>",
        "<|unk|>",
        "<|pad|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|tool_call|>",
        "<|tool_result|>",
        "<|thinking|>",
        "<|json_start|>",
        "<|json_end|>"
    ]
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    tokenizer.train_from_iterator(dataset_iterator(files), trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    raw_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(raw_path)
    
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=raw_path,
        eos_token="<|endoftext|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>"
    )
    hf_tokenizer.save_pretrained(output_dir)
    print(f"    --> Successfully saved custom FIM & ReAct Tokenizer ({len(special_tokens)} special tokens) to {output_dir}")

if __name__ == "__main__":
    train_custom_bpe_tokenizer()
