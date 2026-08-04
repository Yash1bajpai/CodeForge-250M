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

def is_bracket_balanced(code: str) -> bool:
    """
    Universal syntax validator for C++, JS, and multi-language snippets.
    Checks that braces, parentheses, and brackets are balanced.
    """
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    for char in code:
        if char in "({[":
            stack.append(char)
        elif char in ")}]":
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
    return len(stack) == 0

def filter_code_quality(raw_dir: str = "data/raw", filtered_dir: str = "data/filtered"):
    os.makedirs(filtered_dir, exist_ok=True)
    raw_files = glob.glob(os.path.join(raw_dir, "*_raw.jsonl"))
    
    print("--> [Quality Filter] Filtering for quality and syntax...")
    for file_path in raw_files:
        lang = os.path.basename(file_path).split("_")[0]
        out_path = os.path.join(filtered_dir, f"{lang}_filtered.jsonl")
        
        kept, total = 0, 0
        with open(file_path, "r", encoding="utf-8") as in_f, open(out_path, "w", encoding="utf-8") as out_f:
            for line in in_f:
                total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    code = data.get("text", "")
                except Exception:
                    code = line.replace("\\n", "\n")
                    
                lines = code.splitlines()
                if len(lines) < 2 or len(code) < 30:
                    continue
                alnum_count = sum(c.isalnum() for c in code)
                if len(code) == 0 or (alnum_count / len(code)) < 0.25:
                    continue
                
                # Strict bracket/brace balance check only on pure code datasets
                is_pure_code = any(k in lang for k in ["starcoder", "codeparrot", "python", "stack"])
                if is_pure_code and not is_bracket_balanced(code):
                    continue
                
                # Strict Python AST syntax verification on pure code files
                if is_pure_code:
                    if not is_valid_python_ast(code):
                        continue
                        
                out_f.write(json.dumps({"text": code}, ensure_ascii=False) + "\n")
                kept += 1
        print(f"    --> {lang}: Kept {kept:,} / {total:,} quality samples")

if __name__ == "__main__":
    filter_code_quality()
