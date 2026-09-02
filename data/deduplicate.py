import os
import glob
import json
from hashlib import md5

def get_fast_signatures(code: str):
    """
    Computes exact MD5 plus two 64-bit MinHash band buckets for fast O(1) near-duplicate detection.
    """
    normalized = " ".join(code.split())
    exact_hash = md5(normalized.encode("utf-8", errors="ignore")).hexdigest()
    
    words = normalized.split()
    if len(words) < 5:
        return exact_hash, [exact_hash]
        
    shingles = [" ".join(words[i:i+5]) for i in range(len(words) - 4)]
    # Create 2 bands for sub-linear LSH lookup
    band1_hash = md5(" ".join(shingles[::2]).encode("utf-8", errors="ignore")).hexdigest()[:16]
    band2_hash = md5(" ".join(shingles[1::2]).encode("utf-8", errors="ignore")).hexdigest()[:16]
    return exact_hash, [band1_hash, band2_hash]

def deduplicate_dataset(filtered_dir: str = "data/filtered", dedup_dir: str = "data/dedup"):
    os.makedirs(dedup_dir, exist_ok=True)
    filtered_files = glob.glob(os.path.join(filtered_dir, "*_filtered.jsonl"))
    
    print("--> [Fast LSH Deduplication] Removing exact and near-duplicates across shards...")
    seen_exact = set()
    seen_bands = set()
    
    for file_path in filtered_files:
        lang = os.path.basename(file_path).split("_")[0]
        out_path = os.path.join(dedup_dir, f"{lang}_dedup.jsonl")
        
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
                    
                exact_hash, bands = get_fast_signatures(code)
                if exact_hash in seen_exact:
                    continue
                if any(b in seen_bands for b in bands):
                    continue
                    
                seen_exact.add(exact_hash)
                for b in bands:
                    seen_bands.add(b)
                    
                out_f.write(json.dumps({"text": code}, ensure_ascii=False) + "\n")
                kept += 1
        print(f"    --> {lang}: Kept {kept:,} / {total:,} unique samples")

if __name__ == "__main__":
    deduplicate_dataset()
