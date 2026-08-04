import os
import json
import yaml
from datasets import load_dataset
from typing import Dict, List

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

def download_curated_stack(config_path: str = "configs/config_250M.yaml", output_dir: str = "data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    
    target_tokens = 2350000000
    sources = ["starcoder-python", "glaive-function-calling", "codeparrot-clean", "evol-codealpaca", "tiny-textbooks", "commitpackft-python", "code-contests"]
    weights = [0.45, 0.15, 0.12, 0.10, 0.08, 0.05, 0.05]
    
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if "data" in cfg:
                target_tokens = cfg["data"].get("target_tokens", target_tokens)
                sources = cfg["data"].get("languages", sources)
                weights = cfg["data"].get("language_weights", weights)
                
    # Approximate 4 characters per token
    target_chars_per_src = {src: int(target_tokens * 4 * w) for src, w in zip(sources, weights)}
    print(f"--> [Data Pipeline] Starting Production Multi-Source Streaming Download (Target Tokens: {target_tokens:,})...")
    
    hf_mapping = {
        "starcoder-python": ("bigcode/starcoderdata", "python", "content"),
        "glaive-function-calling": ("glaiveai/glaive-function-calling-v2", None, "system_and_chat"),
        "codeparrot-clean": ("codeparrot/codeparrot-clean-train", None, "content"),
        "evol-codealpaca": ("theblackcat102/evol-codealpaca-v1", None, "instruction_output"),
        "tiny-textbooks": ("nampdn-ai/tiny-textbooks", None, "text"),
        "commitpackft-python": ("bigcode/commitpackft", "python", "diff"),
        "code-contests": ("deepmind/code_contests", None, "problem_solution")
    }
    
    for src, max_chars in target_chars_per_src.items():
        print(f"    Fetching {src} (Target chars: {max_chars:,})...")
        repo, subset, mode = hf_mapping.get(src, ("codeparrot/codeparrot-clean-train", None, "content"))
        
        try:
            if subset:
                ds = load_dataset(repo, subset, split="train", streaming=True, token=os.environ.get("HF_TOKEN") or None)
            else:
                ds = load_dataset(repo, split="train", streaming=True, token=os.environ.get("HF_TOKEN") or None)
        except Exception as e:
            print(f"    [Warning] Could not stream {repo} ({e}). Skipping source to avoid fallback contamination.")
            continue
            
        out_file = os.path.join(output_dir, f"{src}_raw.jsonl")
        char_count = 0
        with open(out_file, "w", encoding="utf-8") as out_f:
            for sample in ds:
                code = ""
                if mode == "system_and_chat":
                    system = sample.get("system", "")
                    chat = sample.get("chat", "")
                    code = f"{system}\n\n{chat}".strip()
                elif mode == "instruction_output":
                    inst = sample.get("instruction", "")
                    out = sample.get("output", "")
                    code = f"### Instruction:\n{inst}\n\n### Response:\n{out}".strip()
                elif mode == "diff":
                    buggy = sample.get("old_contents") or sample.get("buggy") or ""
                    fixed = sample.get("new_contents") or sample.get("fixed") or ""
                    code = f"<|fim_prefix|>{buggy}<|fim_middle|>{fixed}".strip()
                elif mode == "problem_solution":
                    desc = sample.get("description", "")
                    sols = sample.get("solutions", {}).get("solution", [])
                    sol = sols[0] if isinstance(sols, list) and sols else str(sols)
                    code = f"### Problem:\n{desc}\n\n### Solution:\n{sol}".strip()
                else:
                    code = sample.get("content") or sample.get("code") or sample.get("text") or ""
                    
                if isinstance(code, str) and len(code) > 20:
                    out_f.write(json.dumps({"text": code}, ensure_ascii=False) + "\n")
                    char_count += len(code)
                if char_count >= max_chars:
                    break
        print(f"    --> Saved {char_count:,} chars (~{char_count//4:,} tokens) for {src} to {out_file}")

if __name__ == "__main__":
    download_curated_stack()
