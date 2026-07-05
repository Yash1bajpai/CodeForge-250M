import os
import ast
import glob

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
    
    print("--> [Quality Filter] Filtering for Python AST + Universal Brace/Paren Balance...")
    for file_path in raw_files:
        lang = os.path.basename(file_path).split("_")[0]
        out_path = os.path.join(filtered_dir, f"{lang}_filtered.jsonl")
        
        kept, total = 0, 0
        with open(file_path, "r", encoding="utf-8") as in_f, open(out_path, "w", encoding="utf-8") as out_f:
            for line in in_f:
                total += 1
                code = line.replace("\\n", "\n")
                lines = code.splitlines()
                if len(lines) < 3 or len(code) / len(lines) > 200:
                    continue
                alnum_count = sum(c.isalnum() for c in code)
                if len(code) == 0 or (alnum_count / len(code)) < 0.30:
                    continue
                
                # Universal Bracket/Brace Balance check for ALL languages (JS, C++, Python, etc.)
                if not is_bracket_balanced(code):
                    continue
                
                # Python AST syntax verification
                if "python" in lang or "stack" in lang or "commit" in lang:
                    if not is_valid_python_ast(code):
                        continue
                        
                out_f.write(line)
                kept += 1
        print(f"    --> {lang}: Kept {kept:,} / {total:,} syntactically valid samples")

if __name__ == "__main__":
    filter_code_quality()
