import os
import json
import yaml
from datasets import load_dataset
from typing import Dict, List

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

def download_curated_stack(config_path: str = "configs/config_250M.yaml", output_dir: str = "data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    
    target_tokens = 2350000000
    sources = ["starcoder-python", "codeparrot-clean", "fineweb-edu", "tiny-textbooks", "evol-codealpaca", "glaive-function-calling", "commitpackft-python"]
    weights = [0.53, 0.21, 0.10, 0.075, 0.05, 0.025, 0.01]
    
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
        "starcoder-python": ("bigcode/starcoderdata", "PARQUET:python", "content"),
        "glaive-function-calling": ("glaiveai/glaive-function-calling-v2", None, "system_and_chat"),
        "codeparrot-clean": ("codeparrot/codeparrot-clean-train", None, "content"),
        "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
        "evol-codealpaca": ("theblackcat102/evol-codealpaca-v1", None, "instruction_output"),
        "tiny-textbooks": ("nampdn-ai/tiny-textbooks", None, "text"),
        "commitpackft-python": ("bigcode/commitpackft", "python", "diff"),
    }

    def stream_parquet_dir(repo, subdir):
        """datasets 3.x removed starcoderdata's per-language configs; stream its
        per-language parquet directory directly. Returns an iterable of rows."""
        import requests
        from datasets import load_dataset
        token = os.environ.get("HF_TOKEN") or None
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        api = f"https://huggingface.co/api/datasets/{repo}/tree/main/{subdir}"
        files = [f["path"] for f in requests.get(api, headers=headers).json()
                 if f["path"].endswith(".parquet")]
        print(f"    [Parquet] {len(files)} parquet files in {repo}/{subdir}")
        for fp in files:
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/{fp}"
            ds = load_dataset("parquet", data_files=url, split="train",
                              streaming=True, token=token)
            for row in ds:
                yield row

    for src, max_chars in target_chars_per_src.items():
        print(f"    Fetching {src} (Target chars: {max_chars:,})...")
        if src not in hf_mapping:
            raise SystemExit(f"ERROR: source '{src}' has no hf_mapping entry. Refusing to fall back to another dataset (run #1 overfitting cause).")
        repo, subset, mode = hf_mapping[src]

        out_file = os.path.join(output_dir, f"{src}_raw.jsonl")
        # RESUME: skip sources already (mostly) downloaded in a previous session
        if os.path.exists(out_file) and os.path.getsize(out_file) >= 0.97 * max_chars:
            print(f"    [Resume] {src} already downloaded ({os.path.getsize(out_file):,} chars). Skipping.")
            continue

        try:
            if isinstance(subset, str) and subset.startswith("PARQUET:"):
                ds = stream_parquet_dir(repo, subset.split(":", 1)[1])
            elif subset:
                ds = load_dataset(repo, subset, split="train", streaming=True,
                                  token=os.environ.get("HF_TOKEN") or None,
                                  trust_remote_code=True)
            else:
                ds = load_dataset(repo, split="train", streaming=True,
                                  token=os.environ.get("HF_TOKEN") or None,
                                  trust_remote_code=True)
        except Exception as e:
            print(f"    [Warning] Could not stream {repo} ({e}). Skipping source to avoid fallback contamination.")
            continue

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
