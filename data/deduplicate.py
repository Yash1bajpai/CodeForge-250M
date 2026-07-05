import os
import glob
from hashlib import md5

def get_shingle_minhash(code: str, num_perm: int = 64, shingle_size: int = 5):
    """
    Computes a lightweight MinHash signature using n-gram character shingles.
    Captures near-duplicates (renamed variables, minor edits, boilerplate) with Jaccard similarity < 0.85.
    """
    words = code.split()
    if len(words) < shingle_size:
        shingles = [code]
    else:
        shingles = [" ".join(words[i:i+shingle_size]) for i in range(len(words) - shingle_size + 1)]
    
    # Compute minhash signature across num_perm hash seeds
    sig = []
    for seed in range(num_perm):
        min_h = float('inf')
        for sh in shingles:
            h = int(md5((f"{seed}_{sh}").encode('utf-8')).hexdigest()[:8], 16)
            if h < min_h:
                min_h = h
        sig.append(min_h)
    return sig

def estimated_jaccard(sig1, sig2):
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)

def deduplicate_dataset(filtered_dir: str = "data/filtered", dedup_dir: str = "data/dedup", jaccard_threshold: float = 0.85):
    os.makedirs(dedup_dir, exist_ok=True)
    filtered_files = glob.glob(os.path.join(filtered_dir, "*_filtered.jsonl"))
    
    print(f"--> [True MinHash Deduplication] Removing near-duplicates (Jaccard >= {jaccard_threshold}) across shards...")
    seen_signatures = []
    
    for file_path in filtered_files:
        lang = os.path.basename(file_path).split("_")[0]
        out_path = os.path.join(dedup_dir, f"{lang}_dedup.jsonl")
        
        kept, total = 0, 0
        with open(file_path, "r", encoding="utf-8") as in_f, open(out_path, "w", encoding="utf-8") as out_f:
            for line in in_f:
                total += 1
                code = line.replace("\\n", "\n")
                sig = get_shingle_minhash(code, num_perm=32)
                
                # Check near-duplicate Jaccard similarity against recent signatures
                is_duplicate = False
                for prev_sig in seen_signatures[-2000:]:  # Windowed check for speed & memory
                    if estimated_jaccard(sig, prev_sig) >= jaccard_threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    seen_signatures.append(sig)
                    out_f.write(line)
                    kept += 1
        print(f"    --> {lang}: True MinHash kept {kept:,} / {total:,} unique samples")

if __name__ == "__main__":
    deduplicate_dataset()
