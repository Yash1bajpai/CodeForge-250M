import glob
import json
import os
from contextlib import ExitStack
from hashlib import sha256

from datasketch import MinHash, MinHashLSH


SHINGLE_SIZE = 5
MINHASH_SEED = 1


def _read_record(line):
    line = line.strip()
    if not line:
        return None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        record = {"text": line.replace("\\n", "\n")}
    if not isinstance(record, dict) or not isinstance(record.get("text"), str):
        return None
    return record if record["text"].strip() else None


def _word_shingles(text):
    words = text.split()
    return {" ".join(words[i:i + SHINGLE_SIZE])
            for i in range(len(words) - SHINGLE_SIZE + 1)}


def _minhash(shingles, num_perm):
    signature = MinHash(num_perm=num_perm, seed=MINHASH_SEED)
    batch = []
    for shingle in shingles:
        batch.append(shingle.encode("utf-8"))
        if len(batch) == 256:
            signature.update_batch(batch)
            batch.clear()
    if batch:
        signature.update_batch(batch)
    return signature


def deduplicate_dataset(filtered_dir: str = "data/filtered", dedup_dir: str = "data/dedup",
                        threshold: float = 0.85, num_perm: int = 128):
    """Greedily retain documents in sorted source-file/record order.

    Exact matching normalizes whitespace. Near matching uses seeded MinHash LSH
    candidates, then verifies their normalized word-five-gram Jaccard similarity.
    LSH can miss similar pairs: this does not establish global corpus uniqueness.
    Documents with fewer than five words receive exact-only deduplication.

    Exact digests, LSH buckets and retained-document offsets grow in RAM with the
    document count. Verification rereads source records rather than keeping all
    text/shingle sets in memory; inputs must not change during a run. One source
    reader per processed file remains open until completion. MinHash batch
    temporaries are limited to 256 shingles, but an individual document's shingle
    set still must fit in RAM.
    """
    if not 0 < threshold < 1:
        raise ValueError("Deduplication threshold must be between 0 and 1")
    if not isinstance(num_perm, int) or num_perm < 16:
        raise ValueError("Deduplication num_perm must be an integer of at least 16")
    filtered_files = sorted(glob.glob(os.path.join(filtered_dir, "*_filtered.jsonl")))
    if not filtered_files:
        raise ValueError(f"No *_filtered.jsonl inputs found in {filtered_dir}")

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    os.makedirs(dedup_dir, exist_ok=True)
    stats = {
        "parameters": {
            "algorithm": "datasketch.MinHashLSH with exact candidate Jaccard verification",
            "threshold": threshold,
            "num_perm": num_perm,
            "seed": MINHASH_SEED,
            "shingle_size_words": SHINGLE_SIZE,
            "normalization": "whitespace split/join; case preserved",
            "exact_hash": "sha256",
            "short_documents": "exact-only below five words",
            "lsh_bands": lsh.b,
            "lsh_rows_per_band": lsh.r,
            "probabilistic_candidate_recall": True,
        },
        "totals": {"input_records": 0, "invalid_records": 0, "exact_removed": 0,
                   "near_removed": 0, "kept": 0},
        "sources": {},
    }
    seen_exact = set()
    retained_offsets = []
    source_readers = []
    print("--> [MinHash LSH Deduplication] Removing exact and verified near-duplicates...")
    with ExitStack() as stack:
        for source_index, file_path in enumerate(filtered_files):
            source = os.path.basename(file_path).removesuffix("_filtered.jsonl")
            out_path = os.path.join(dedup_dir, f"{source}_dedup.jsonl")
            counts = dict.fromkeys(stats["totals"], 0)
            stats["sources"][source] = counts
            source_readers.append(stack.enter_context(open(file_path, "rb")))
            with open(file_path, "rb") as in_f, open(out_path, "w", encoding="utf-8") as out_f:
                while True:
                    offset = in_f.tell()
                    line = in_f.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    counts["input_records"] += 1
                    record = _read_record(line.decode("utf-8"))
                    if record is None:
                        counts["invalid_records"] += 1
                        continue
                    normalized = " ".join(record["text"].split())
                    exact_hash = sha256(normalized.encode("utf-8")).digest()
                    if exact_hash in seen_exact:
                        counts["exact_removed"] += 1
                        continue
                    seen_exact.add(exact_hash)

                    shingles = _word_shingles(normalized)
                    signature = _minhash(shingles, num_perm) if shingles else None
                    near_duplicate = False
                    if signature is not None:
                        for candidate_id in sorted(lsh.query(signature)):
                            candidate_source, candidate_offset = retained_offsets[candidate_id]
                            reader = source_readers[candidate_source]
                            reader.seek(candidate_offset)
                            candidate = _read_record(reader.readline().decode("utf-8"))
                            if candidate is None:
                                raise RuntimeError("Deduplication input changed during processing")
                            other = _word_shingles(candidate["text"])
                            intersection_size = len(shingles.intersection(other))
                            union_size = len(shingles) + len(other) - intersection_size
                            if intersection_size / union_size >= threshold:
                                near_duplicate = True
                                break
                    if near_duplicate:
                        counts["near_removed"] += 1
                        continue

                    if signature is not None:
                        candidate_id = len(retained_offsets)
                        retained_offsets.append((source_index, offset))
                        lsh.insert(candidate_id, signature)
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts["kept"] += 1
            for key, value in counts.items():
                stats["totals"][key] += value
            print(f"    --> {source}: Kept {counts['kept']:,} / {counts['input_records']:,} records")

    if not stats["totals"]["kept"]:
        raise ValueError(f"No nonempty text records found in {filtered_dir}")
    with open(os.path.join(dedup_dir, "dedup_stats.json"), "w", encoding="utf-8") as stats_f:
        json.dump(stats, stats_f, indent=2, ensure_ascii=False)
        stats_f.write("\n")
    return stats


if __name__ == "__main__":
    deduplicate_dataset()
