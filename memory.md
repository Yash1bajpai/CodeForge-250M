# CodeForge-250M — Project Memory & State

*Living technical record. Read this first; it supersedes any older narrative.*

---

## 1. Project Identity

* **Model:** `CodeForge-250M`
* **Architecture:** LLaMA-2 style decoder-only. 16 layers, hidden 1024, FFN 2816, 16 attention heads, 4 KV heads (GQA), RoPE, RMSNorm, SwiGLU, untied embeddings. SDPA attention.
* **Precision:** BF16 on Ampere+ / FP16 + GradScaler on Turing.
* **Purpose:** Local edge coding assistant + inference backend for Nexus-Agent (ReAct loop, FIM infill, structured tool calls).
* **Platform:** Lightning AI Studio over SSH (`ssh.lightning.ai`), RTX PRO 6000 in AWS Spot mode (~$2.05/hr).

---

## 2. RUN #1 — FAILED BY OVERFITTING (post-mortem)

The first run reached step 102,546 / ~1.02B cumulative tokens with a final loss of
**0.0153 (perplexity 1.02)**. This was previously logged as "world-class syntax
mastery." **That interpretation was wrong.** A perplexity of 1.02 on a 250M-parameter
model means the model was reproducing its training data from memory, not generalising.
The run is discarded. Nothing from it is reused.

**Root cause — three compounding faults:**

1. **Effective batch was 128x smaller than designed.** 1,019,834,368 tokens across
   102,546 steps = **~9,945 tokens/step**, against a configured 524,288 tokens/step.
   Gradient accumulation was not in effect; the run used roughly batch 16 x 1 seq.
   Tiny batches on repeated data drive memorisation fast.
2. **Too few unique tokens, cycled many times.** `train.py` wraps the dataloader in an
   infinite `while True: for b in loader` generator that silently re-iterates the same
   shards forever. With an estimated 50-200M unique tokens available, 1.02B tokens
   processed means **~5 to 20 epochs over the same corpus**. Loss collapse was
   inevitable and was mistaken for progress.
3. **Success was measured only by falling training loss.** No held-out validation set
   and no HumanEval/MBPP gate ran during training, so there was no signal that could
   have distinguished learning from memorisation.

**Lesson:** for a 250M model, target ~1 epoch over a genuinely large unique corpus.
Training loss alone cannot detect overfitting — held-out loss can.

---

## 3. RUN #2 — FRESH FROM STEP 0 (current)

Old checkpoints have been removed. Training restarts with randomly initialised weights.

**Target:** 2.35B **unique** tokens, single pass.
`4,482 steps x 256 seqs x 2,048 tokens = 2,349,858,816 tokens` — matches
`target_tokens: 2350000000`. Verified consistent.

**Batch:** `batch_size_per_device 16 x gradient_accumulation_steps 16 = 256 sequences
= 524,288 tokens/step`. Verified consistent with `total_batch_size_sequences: 256`.

**LR schedule:** cosine, max 6.0e-4 -> min 6.0e-5, 500 warmup steps (~11% of run).

**Data curriculum (weights verified to sum to exactly 1.0):**

| source | weight | HF dataset |
|---|---|---|
| starcoder-python | 0.53 | `bigcode/starcoderdata` (python) |
| codeparrot-clean | 0.21 | `codeparrot/codeparrot-clean-train` |
| fineweb-edu | 0.10 | **NOT MAPPED — see blocker B1** |
| tiny-textbooks | 0.075 | `nampdn-ai/tiny-textbooks` |
| evol-codealpaca | 0.05 | `theblackcat102/evol-codealpaca-v1` |
| glaive-function-calling | 0.025 | `glaiveai/glaive-function-calling-v2` |
| commitpackft-python | 0.01 | `bigcode/commitpackft` (python) |

**Pipeline:** `download_stack.py` -> `data/raw/*_raw.jsonl` -> `filter_quality.py` ->
`data/filtered/*_filtered.jsonl` -> `deduplicate.py` -> `data/dedup/*_dedup.jsonl` ->
`train_tokenizer.py` -> `data/tokenizer/` -> `tokenize_dataset.py` ->
`data/tokenized/shard_*.pt`. Directory handoffs verified consistent; no stage points at
a stale directory from run #1.

---

## 4. BLOCKERS — must be fixed before launching run #2

Found by audit of the code as committed at `88dede2`. Ordered by severity.

**B1 — `fineweb-edu` silently becomes duplicate `codeparrot-clean`.**
`download_stack.py:28-36` defines `hf_mapping` without a `fineweb-edu` key. Line 40
falls back to `("codeparrot/codeparrot-clean-train", None, "content")` for any unmapped
source. `fineweb-edu` carries **weight 0.10 = ~235M tokens**, which would be downloaded
a second time from codeparrot and written to `fineweb-edu_raw.jsonl`. Cross-file
dedup in `deduplicate.py` would then discard most of it, leaving the run short of
tokens and biased toward codeparrot. **This is the exact duplicate-data mechanism that
caused run #1's overfitting.** Either add the real mapping
(`HuggingFaceFW/fineweb-edu`, field `text`) or remove the source and redistribute its
weight. Also: `hf_mapping` contains an unused `code-contests` entry not present in the
config.

**B2 — `train.py` silently resumes from a checkpoint unless `--from_scratch` is passed.**
`train.py:157` — `if os.path.exists(latest_path) and not args.from_scratch:` loads
`latest_checkpoint.pt`, restores model + optimizer state, and sets
`start_step = ckpt['step']` (line 162). `scripts/start_training.sh:11` invokes
`train.py` with **no arguments**, so if any `latest_checkpoint.pt` is present in
`checkpoints/CodeForge-250M/` on the Lightning studio, the "fresh" run resumes the
overfitted weights instead. `start_training.sh` also hardcodes the absolute path
`/teamspace/studios/this_studio/CodeForge-250M/training/train.py`. Fix: pass
`--from_scratch`, and confirm the remote checkpoint directory is empty before launch.

**B3 — the committed tokenizer is missing 5 of its 11 special tokens.**
`train_tokenizer.py:34-46` declares `<|endoftext|> <|unk|> <|pad|> <|fim_prefix|>
<|fim_middle|> <|fim_suffix|> <|tool_call|> <|tool_result|> <|thinking|>
<|json_start|> <|json_end|>`. The committed `data/tokenizer/tokenizer.json` contains
only the first **6** (IDs 0-5). The five ReAct/tool tokens are absent, so BPE will
split them into ordinary sub-word pieces — defeating the structured tool-calling and
`<|thinking|>` trace design in `serving/nexus_bridge.py`. The tokenizer must be
retrained on the new dedup corpus before `tokenize_dataset.py` runs. Vocab size is
correct at 32,000, and `<|unk|>` (1) and `<|pad|>` (2) are correctly distinct.

**B4 — `ShardedCodeDataset` loads the entire corpus into host RAM.**
`train.py:47-61` globs every `shard_*.pt` and `extend`s them into one Python list.
For a full epoch: `1,147,392 sequences x 2,048 tokens x 8 bytes (int64) = ~17.5 GB`
of host RAM, before DataLoader worker copies (`num_workers=2`, `pin_memory=True`).
This will OOM or thrash on the studio. Convert to lazy per-shard loading, memory-map,
or store as `int32`/`uint16` (vocab is 32k, so 16 bits suffice — a 4x reduction).

**B5 — checkpoint interval regressed to 100 steps.**
`config_250M.yaml: checkpointing.save_steps: 100` gives `4482/100 = 44` checkpoints at
~2.8 GB each = **~123 GB of checkpoints**. Run #1 hit a 180 GB disk-bloat crisis and
the fix was saving every 1,000 steps; the config has since reverted to 100. Set
`save_steps: 1000` (~4-5 checkpoints, and ~6.5 min of Spot exposure, which is the
protection the 1,000-step rule was designed for).

**B6 — the AST filter destroys `commitpackft-python`.**
`filter_quality.py:59` sets `is_pure_code = True` when the source name contains
`python`, which matches `commitpackft-python`. But `download_stack.py:67` formats that
source as `<|fim_prefix|>{buggy}<|fim_middle|>{fixed}`, which is not parseable Python.
`ast.parse` will reject essentially every sample, silently zeroing this source. Weight
is only 0.01, but the diff-editing capability it provides is the whole point of
including it. Exclude FIM-formatted sources from AST validation.

**B7 — no validation split, so overfitting remains undetectable.**
`config_250M.yaml` declares `val_split: "validation"` but nothing in `train.py` builds
or evaluates a held-out set; only training loss is printed. Given that undetected
memorisation is precisely what killed run #1, hold out ~0.5% of shards and log
validation loss at each checkpoint. Diverging train and val loss is the signal to stop.

**B8 — `deepspeed_config: "configs/ds_zero2.json"` does not exist.**
`configs/` contains only the three model YAMLs. Harmless while `train.py` uses plain
PyTorch AMP (it never reads the key), but remove it or create the file so it does not
mislead later.

**B9 — both data stages silently fall back to raw, unfiltered data.**
`train_tokenizer.py:26` falls back to `data/raw/*.jsonl` when `data/dedup/*_dedup.jsonl`
is empty, and `tokenize_dataset.py:36` does the same. If the filter or dedup stage fails
or is skipped, the tokenizer trains on — and the shards are built from — raw data that
never passed quality filtering or deduplication. That is duplicate-heavy input feeding
straight into training, silently. `train.py:64-66` has the same fallback chain
(`data/dedup` -> `data/raw`). Make these stages fail loudly instead of degrading.

**Minor:** `train.py:270` reads `step` after the loop; if `max_steps <= start_step`
the loop never runs and this raises `NameError`. `tokenize_dataset.py:73` computes the
final partial shard index as `(chunk_count // shard_size) + 1`, an off-by-one against
the in-loop naming at line 67.

---

## 5. Verified as correct (no action needed)

* Weight init: `init_model_weights` is called on the from-scratch path
  (`train.py:170`) and applies LLaMA-2 residual scaling `1/sqrt(2*num_layers)` to
  `o_proj` and `down_proj` (`init_weights.py:14, 22-24`).
* Loss shift: `architecture.py:150-152` shifts logits/labels correctly, so passing
  `x == y` from the dataset is not an off-by-one bug.
* Gradient clipping is applied at `max_grad_norm: 1.0` (`train.py:218`).
* `GradScaler` is correctly enabled only for FP16 and is a no-op under BF16.
* RMSNorm computes in FP32 and casts back (`architecture.py:13-17`) — numerically safe
  under mixed precision.
* LR schedule code matches config values exactly (warmup 500, max_steps 4482,
  6.0e-4 -> 6.0e-5).
* FIM rate is 0.50 in `tokenize_dataset.py:8`, matching the StarCoder/DeepSeek standard.
* `STOP_AND_SAVE` file trigger works and removes the flag file after firing
  (`train.py:246-267`).
* `callbacks.py:63-71` `load_checkpoint_for_resume` is dead code — never called from
  `train.py`, returns 0 unconditionally. No risk, but delete it to avoid confusion.

---

## 6. Budget

Remaining Lightning credit: **~$14.17**. At RTX PRO 6000 Spot ($2.05/hr, ~300M
tokens/hr) the 2.35B-token run costs roughly **$16** and takes ~8 hours — slightly
over budget. Plan for a checkpointed multi-session run, or use free T4 hours for the
data-prep stages and reserve paid GPU time for training only.

---

## 7. Repository layout

`CodeForge-250M_win/` is the single canonical working copy (remote:
`github.com/Yash1bajpai/CodeForge-250M`). A stale divergent clone, ~69 one-off driver
scripts from run #1, and the run #1 training log now live in `../_ATTIC/` — kept for
reference (the SSH and HF-upload logic in those drivers is worth cannibalising) but
outside the repo.
