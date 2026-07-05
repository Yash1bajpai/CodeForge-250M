import os
import glob
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast

def train_custom_bpe_tokenizer(data_dir: str = "data/dedup", output_dir: str = "data/tokenizer", vocab_size: int = 32000):
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(data_dir, "*_dedup.jsonl"))
    if not files:
        files = glob.glob(os.path.join("data", "raw", "*.jsonl"))

    print(f"--> [Tokenizer] Training 32k BPE Tokenizer with distinct <|unk|> and <|pad|> on {len(files)} files...")
    
    # Cleanly separate UNK and PAD tokens!
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    special_tokens = [
        "<|endoftext|>",
        "<|unk|>",
        "<|pad|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>"
    ]
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    tokenizer.train(files, trainer)
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
    print(f"    --> Successfully saved custom FIM Tokenizer with distinct UNK/PAD to {output_dir}")

if __name__ == "__main__":
    train_custom_bpe_tokenizer()
