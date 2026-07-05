import os
from datasets import load_dataset
from typing import Dict, List

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

def download_curated_stack(output_dir: str = "data/raw", target_tokens: int = 10000000):
    os.makedirs(output_dir, exist_ok=True)
    sources = ["the-stack-v2", "commitpackft", "evol-codealpaca", "stackoverflow-py", "cosmopedia-edu"]
    weights = [0.72, 0.07, 0.09, 0.06, 0.06]
    
    target_chars_per_src = {src: int(target_tokens * 4 * w) for src, w in zip(sources, weights)}
    print(f"--> [Data Pipeline] Starting 40MB Smoke Test Streaming Download (Total Target Tokens: {target_tokens:,})...")
    
    hf_mapping = {
        "the-stack-v2": ("flytech/python-codes-25k", None),
        "commitpackft": ("google/code_x_glue_cc_code_refinement", "python"),
        "evol-codealpaca": ("iamtarun/python_code_instructions_18k_alpaca", None),
        "stackoverflow-py": ("sahil2801/CodeAlpaca-20k", None),
        "cosmopedia-edu": ("HuggingFaceTB/cosmopedia", "web_samples_v2")
    }
    
    for src, max_chars in target_chars_per_src.items():
        print(f"    Fetching {src} (Target chars: {max_chars:,})...")
        repo, subset = hf_mapping.get(src, ("flytech/python-codes-25k", None))
        
        try:
            if subset:
                ds = load_dataset(repo, subset, split="train", streaming=True, token=os.environ["HF_TOKEN"])
            else:
                ds = load_dataset(repo, split="train", streaming=True, token=os.environ["HF_TOKEN"])
        except Exception as e:
            print(f"    [Fallback] Could not stream {repo} ({e}). Using iamtarun/python_code_instructions_18k_alpaca fallback...")
            ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True, token=os.environ["HF_TOKEN"])
            
        out_file = os.path.join(output_dir, f"{src}_raw.jsonl")
        char_count = 0
        with open(out_file, "w", encoding="utf-8") as out_f:
            for sample in ds:
                code = sample.get("whole_func_string") or sample.get("code") or sample.get("content") or sample.get("instruction", "") + "\n" + sample.get("output", "") or sample.get("buggy", "") + "\n" + sample.get("fixed", "")
                if len(code) > 20:
                    out_f.write(code.replace("\n", "\\n") + "\n")
                    char_count += len(code)
                if char_count >= max_chars:
                    break
        print(f"    --> Saved {char_count:,} chars for {src} to {out_file}")

if __name__ == "__main__":
    download_curated_stack()
