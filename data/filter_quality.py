import os
import ast
import glob
import json

def is_valid_python_ast(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

def filter_code_quality(raw_dir: str = "data/raw", filtered_dir: str = "data/filtered"):
    os.makedirs(filtered_dir, exist_ok=True)
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*_raw.jsonl")))
    if not raw_files:
        raise ValueError(f"No *_raw.jsonl inputs found in {raw_dir}")

    total_kept = 0
    print("--> [Quality Filter] Filtering for quality and syntax...")
    for file_path in raw_files:
        source = os.path.basename(file_path).removesuffix("_raw.jsonl")
        out_path = os.path.join(filtered_dir, f"{source}_filtered.jsonl")
        
        kept, total = 0, 0
        with open(file_path, "r", encoding="utf-8") as in_f, open(out_path, "w", encoding="utf-8") as out_f:
            for line in in_f:
                total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
                        continue
                    code = data["text"]
                except json.JSONDecodeError:
                    code = line.replace("\\n", "\n")
                    data = {"text": code}
                    
                lines = code.splitlines()
                if len(lines) < 2 or len(code) < 30:
                    continue
                alnum_count = sum(c.isalnum() for c in code)
                if len(code) == 0 or (alnum_count / len(code)) < 0.25:
                    continue
                
                # AST validation applies only to pure Python sources, not prose,
                # instruction/tool records or already FIM-formatted documents.
                is_pure_code = any(
                    source.startswith(prefix)
                    for prefix in ("starcoder", "codeparrot", "python", "stack")
                )
                is_fim_source = (
                    "commitpackft" in source
                    or "fim" in source
                    or any(token in code for token in (
                        "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"
                    ))
                )
                if is_pure_code and not is_fim_source and not is_valid_python_ast(code):
                    continue

                out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
                kept += 1
        total_kept += kept
        print(f"    --> {source}: Kept {kept:,} / {total:,} quality samples")

    if not total_kept:
        raise ValueError(f"No records passed quality filtering in {raw_dir}")

if __name__ == "__main__":
    filter_code_quality()
