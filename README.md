# CodeForge-250M

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-SDPA-ee4c2c)](https://pytorch.org/)

CodeForge-250M is a roughly 246M-parameter decoder-only code model trained from scratch. The goal is to support code completion, FIM editing and tool calling for nexus-agent and Vision. Those are development goals, not validated capabilities of the current checkpoint.

**Full project plan and another-PC handoff:** [`CODEFORGE_PLAN.md`](CODEFORGE_PLAN.md).

## Current state

- Stage A is complete at step **2,898**, covering approximately **1.52B tokens in the intended single-pass schedule**. This is not a claim that every token is unique or that the run was free of legacy training defects.
- The measured checkpoint is in `Yash1bajpai/CodeForge-250M-rotation` at revision `2542b76813c0323db7325a0dfed56431964edbf5`. That checkpoint repository is private; access is required to reproduce the evaluation.
- Real KV-cache inference and HumanEval/MBPP evaluation are implemented in [`evaluation/codeforge_eval.py`](evaluation/codeforge_eval.py).
- The checkpoint is not yet a reliable coding or tool-calling model. Plain completion can leak FIM tokens or stop early.
- [`serving/nexus_bridge.py`](serving/nexus_bridge.py) is still a placeholder: it returns hardcoded text, does not load model weights and does not implement grammar-constrained decoding.
- Stage B has not been done. Tool-use SFT has been tried in small experiments (see below); none is promoted. The 10B-token goal is a future milestone, not the amount already trained.

## Fresh restart preparation (October 25, 2026)

`configs/config_250M_fresh.yaml` defines a new random-weight 245,924,864-parameter
run. The target budget is 10B non-padding input tokens: 8.5B foundation, 1B
code-instruction and 0.5B tool-instruction. These are staged planning budgets, not
claims of available unique data or model quality. The executable foundation
schedule is 16,221 nominal updates at 256 sequences of 2,048 tokens. The trainer
now has a foundation-only production-window consumer and cumulative real-input
accounting, verified on tiny CPU fixtures only. Instruction loss masking and
full-scale throughput remain gates. Existing checkpoints are not initialization
sources and are not overwritten.

**Model training is not authorized by data preparation.** The fresh config sets
`require_explicit_approval: true`; the training entrypoint exits before CUDA
setup or log/checkpoint creation unless `CF_TRAINING_APPROVED=1` is explicitly
provided. Notify the owner before any GPU pilot or training launch. This flag
is an operator approval boundary, not proof that the other readiness gates pass.

The preparation entrypoint is `python -m data.prepare_fresh`. The bounded
`configs/fresh_data_pilot.yaml` uses pinned public CodeParrot and FineWeb-Edu
revisions, explicit raw-character caps, an allowlist of per-record code licenses,
and retained repository/path/license provenance. Its source mixture is only a
CPU pilot, not the production foundation recipe. It excludes high-specificity
32-token windows from pinned HumanEval/MBPP text/code/tests, performs cross-source
deduplication, and splits documents deterministically before fitting the tokenizer.
This exclusion does not establish complete benchmark decontamination.

The candidate tokenizer has 32k entries and 11 verified special tokens. Train and
validation packing share its recorded identity. Shards have matching valid-length
sidecars, real/padding/discarded token counts, and SHA-256 manifests. The completed
pilot is a **candidate**: it does not promote a production tokenizer, initialize
model weights, or demonstrate coding/tool-use quality. MinHash candidate recall
is probabilistic and its current corpus index uses RAM; 10B-scale deduplication
and loading need a measured bounded-memory/data-window plan.

The verified owner CPU build is
[`codeforge-fresh-cpu-preparation-139cbd2a`](https://www.kaggle.com/code/yashbajpai2027/codeforge-fresh-cpu-preparation-139cbd2a),
version 1. Preparation itself took 139.7 seconds (not total notebook startup).
It retained 3,272 training documents and 61 validation documents:

| Split | Stored real token IDs | Sequences | EOS padding IDs |
|---|---:|---:|---:|
| Train | 6,787,976 | 3,315 | 1,144 |
| Validation | 182,325 | 90 | 1,995 |

These are packed real IDs, not optimizer-consumed tokens or quality scores.
The build discarded no one-token tails. All 24 output files (45,759,067 bytes)
were downloaded to `G:\My Drive\codeforge-storage\datasets\fresh-20261025-pilot01`;
local readback verified the backup checksums, producer shard/sidecar hashes,
source-bundle identity, 32k/11-token vocabulary and disjoint exact document hashes.
On October 26, 2026, the phone independently recovered these outputs and verified
the Drive copies by downloading them: **24 matching files, zero differences**.

The historical pilot launcher below remains available for an explicitly requested pilot. The current authorized production preparation uses the Kaggle-only recipe described next.

```powershell
python scripts/push_cpu_prep.py --report "G:\My Drive\codeforge-storage\logs\fresh-pilot-launch-new.json"
```

The launcher embeds a hash-recorded snapshot of the local preparation source;
it does not depend on unpublished changes being on GitHub. GPU/TPU are disabled,
dependencies are pinned, and the pilot is bounded to one hour. No token is
embedded. `kaggle/check_hf_secret.py` separately records whether Kaggle Secrets
can supply `HF_TOKEN` and pass a private-repository upload/readback check.
Public-source preparation can proceed without that secret; GPU training cannot
proceed until durable checkpoint credentials are verified. Enable the `HF_TOKEN`
secret through the notebook's Add-ons/Secrets UI, never in notebook source/chat.
The [official CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#using-secrets-in-kernels)
provides no CLI support for configuring Secrets. The CPU readiness check reached
public Hugging Face with HTTP 200 but failed in Kaggle's secret backend
(`ConnectionError`); it did not authenticate or verify checkpoint writes from
Kaggle. This is still a pre-training blocker, not a successful credential gate.

## Kaggle-only production preparation (October 26, 2026)

The PC is a controller only. Dataset processing, tokenizer fitting, tokenization,
packing, training and evaluation run on Kaggle. CPU preparation is authorized;
model training is not. The phone setup is preserved but is not the active workflow.

The immutable source revisions and license allowlists are in
[`configs/fresh_data_production.yaml`](configs/fresh_data_production.yaml).
The foundation recipe replaces gated StarCoder data with public multilingual
GitHub code; `configs/config_250M_fresh.yaml` records the same source names.

| Source/subset | Stage | Training-input allocation |
|---|---|---:|
| [CodeParrot GitHub code](https://huggingface.co/datasets/codeparrot/github-code) | Foundation | 5.950B |
| [CodeParrot clean Python](https://huggingface.co/datasets/codeparrot/codeparrot-clean-train) | Foundation | 1.275B |
| [FineWeb-Edu 10BT sample](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Foundation | 1.275B |
| [OpenCoder diverse instructions](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage1) | Code instruction | 0.800B |
| OpenCoder real-user instructions | Code instruction | 0.200B |
| [Glaive function calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Tool instruction | 0.300B |
| [Hermes function calling](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | Tool instruction | 0.200B |

**These are allocations, not available unique-data counts.** In particular,
Glaive/Hermes together contain only about 320MB of source JSON. They cannot
establish 0.5B unique tool tokens. Preparation does not repeat data to fill a
shortfall. Measured retained counts must determine a later training schedule;
more verified execution/Nexus/Vision examples remain a capability requirement.
Public function-call traces are not proof that our model can execute or test code.

Code records require approved per-record repository-license labels; unknown and
copyleft labels are excluded. FineWeb-Edu uses its declared ODC-BY license,
OpenCoder MIT, Glaive/Hermes Apache-2.0. Attribution/provenance is retained with
every admitted document. These producer labels are not an independent per-file
license or privacy audit.

The production engine freezes one 32K byte-level BPE tokenizer from an 80M-character
train-only sample: 56M multilingual code, 12M clean Python, 12M educational prose.
Validation repositories are held out before fitting; validation documents matching
tokenizer-sample exact hashes or sampled repeated blocks are rejected.
It uses disk-backed global normalized-exact deduplication plus eight sampled
256-word repeated blocks per long document. This conservative filter is not
exhaustive near-deduplication. Pinned HumanEval/MBPP lexical-window exclusion
continues to apply; complete benchmark decontamination is not claimed.

Code-only FIM is deterministic. Chat/tool data is never FIM-transformed.
Instruction conversations exceeding 2,048 tokens are counted and rejected,
not silently truncated. Shards include `uint16` IDs, real-length sidecars and
assistant-only loss masks. User, system, tool-result and padding tokens are not
supervised instruction targets. Actual input, valid, padding, discarded and
supervised-target counts are separate; a whole-document budget overshoot is
bounded by the configured document-size cap.

From the PC, submit a private CPU session with a new report filename:

```powershell
python scripts/push_cpu_prep.py --config configs/fresh_data_production.yaml --report ../.watch/fresh-production-launch-new.json
```

`--verify-only` runs the contract suite and live source-format preflight on
Kaggle. `--tests-only` isolates the contract suite without corpus downloads,
tokenizer fitting, preparation or the notebook-secret probe. Normal sessions
run the suite and source preflight before preparing data, then stop cleanly
at four hours, roughly 100M input tokens, or the disk-space floor.
`--session-max-tokens` can reduce a first window without changing the pinned
recipe/checkpoint identity. These are bounded windows, not a single notebook
claiming to prepare all 10B tokens. No GPU/TPU or model-training process is launched.

The original source-schema-corrected submission,
[`codeforge-fresh-cpu-prep-534125d9`](https://www.kaggle.com/code/yashbajpai2027/codeforge-fresh-cpu-prep-534125d9),
prepared a bounded 5M-input window but ended ERROR after its completed engine
failed to exit and hit the outer 15,500-second timeout. Its tokenizer and all ten
windows were separately audited on Kaggle; its old-format state/shards remain
unpromoted. All 93 saved files were preserved with verified download/readback.
The two earlier submissions were superseded and deleted after discovering the
raw GitHub parquet schema: `content`/`path`, with language derived from extension,
rather than the old dataset loader's advertised `code`/`language` output columns.

The PC execution-refusal guard was exercised before output creation. Kaggle CPU
regression run [`codeforge-fresh-cpu-prep-5eb73756`](https://www.kaggle.com/code/yashbajpai2027/codeforge-fresh-cpu-prep-5eb73756)
passed **71 tests in 11.752 seconds**, after six failing-before regressions were
recorded and fixed. These cover space-separated Glaive turns, complete parallel
tool results, assistant-final conversations, interrupted checkpoint recovery,
pipeline identity and interrupted tokenizer publication without refitting.
The live preflight [`codeforge-fresh-cpu-prep-ca3166c4`](https://www.kaggle.com/code/yashbajpai2027/codeforge-fresh-cpu-prep-ca3166c4)
also passed 71 tests and admitted an example from each of the seven pinned
sources. Downloaded reports matched the submitted source snapshot.
Those earlier checks did not prove real-source process shutdown. A bounded
diagnostic reproduced HF range I/O surviving main return. The fix disables
Parquet background prebuffering and explicitly closes source iterators, sampling
resources and session streams before publishing completion; cleanup errors do
not replace an existing processing error.

Corrected [`codeforge-recovered-cpu-prep-2d372b23/1`](https://www.kaggle.com/code/yashbajpai2027/codeforge-recovered-cpu-prep-2d372b23)
is COMPLETE: **76 tests passed in 14.882s**, all seven live-source preflights
passed, the original frozen tokenizer was adopted unchanged, and real
preparation produced 5,045,576 non-padding input IDs. The actual CLI subprocess
exited normally and the outer worker finished; no timeout was suppressed.
These are prepared train-plus-validation IDs, not completed training tokens.
See `CODEFORGE_PLAN.md` for the real-resume/backup result and current notebook.

The real resume `codeforge-fresh-cpu-prep-d4d37022/1` also completed normally
with 76 tests passing. A separate Kaggle audit verified 20 windows, 8,330 distinct
normalized documents, **10,065,678 train input IDs + 101,480 validation IDs**,
unchanged tokenizer/cursors/dedup and four exact byte/control round trips.
The complete 184-file chain was separately backed up to private dataset
`yashbajpai2027/codeforge-fresh-prepared-w20-20261026/1`; another notebook read
back every length/SHA-256. Mounted Drive copies also match, but Drive cloud sync
is unverified; its browser requires sign-in and no login was attempted.

Next CPU preparation `codeforge-fresh-cpu-prep-fe52fd65/1` is RUNNING with a
100M additional-input-token cap, resuming the completed `d4d37022/1` state.
Its additional counts are not yet verified. This is data preparation, not
GPU/model training, and it does not prove the complete 10B corpus is available.

Each committed window has attributed JSONL, hashed shards and an independently
read-back manifest. Checkpoint format v2 commits state, duplicate indexes,
source cursors and recovery metadata in one SQLite transaction. It validates
recipe, preparation-code, Python/package and frozen-tokenizer identities;
recovers missing/stale receipts and hot journals; and preserves the last
committed window on exceptions while rethrowing the original failure.

Resume a completed saved output using
`--resume-kernel yashbajpai2027/<completed-preparation-notebook>` plus the same
config and a fresh report path. The launcher attaches its exact saved version
and pins its Kaggle runtime image. Only the checkpoint database/tokenizer is
copied; earlier windows remain in their parent notebook outputs. The parent
receipt records the notebook version and checkpoint/manifest hashes.
Back up all parent windows, not only the newest one.

A separate-notebook synthetic fixture passed:
`codeforge-cpu-resume-smoke-655fde13/1` → `codeforge-cpu-resume-smoke-1dd18381/1`.
The tokenizer stayed unchanged, the cursor advanced from row 2 to 4, one
duplicate was skipped and the accepted-document total advanced from 2 to 3.
Both output sets were downloaded and checksum/readback verified, then copied
and read back on the mounted Drive. Independent cloud sync is not verified.
This is fixture/provider proof, not production-data resume or token counts.

Kaggle rejects ERROR notebooks as direct inputs; the controller refuses these
before submission. The verified original output was published as the private
recovery dataset `yashbajpai2027/codeforge-prep-recovery-534125d9-20261026/1`.
`kaggle_drive_backup.py` now preserves saved COMPLETE or ERROR output sets while
still refusing active notebooks; this does not make ERROR notebooks attachable.

For this explicit recovery, `--tokenizer-dataset <owner/slug/version>` verifies
the sample/recipe hashes, 32K vocabulary, special tokens, byte alphabet and frozen
tokenizer identity on Kaggle, then imports only tokenizer/sample files into
fresh v2 state. It is mutually exclusive with resume and diagnostic modes.
Old SQLite state, cursors, windows and token totals are never imported.
The recovered candidate identity is `3b06dbefb2abaa646568722ef799ad20ad2c7e4ba2f195583cb29e1a52b0532e`.
Use `--resume-kernel` for subsequent windows, not another adoption or BPE fit.

Kaggle saved outputs are the immediate persistence boundary. This launcher does
not automatically upload production outputs to Drive or Hugging Face. Existing
verified backup commands remain separate; do not claim cloud readback before it
has happened. The Kaggle `HF_TOKEN` checkpoint-write gate remains required before
model training; instruction stages additionally require an SFT loss-mask consumer.


## Foundation production consumer

`training/train.py` accepts `--production-root ROOT --stage foundation`, where
ROOT is a preparation output containing `preparation-manifest.json`, or one
committed window containing `manifest.json`. This does not authorize a training
launch. Without `--production-root`, the legacy flat `data/tokenized/shard_*.pt`
selection and deterministic holdout remain available.

- Preparation roots select only pinned `owned_windows`; parent runs and
  uncommitted/unlisted windows are never discovered automatically.
- `train` and `val` use the producer's existing `train`/`validation` membership,
  not a new holdout. Missing selected splits fail; `CF_VAL_DIR` cannot override
  production splits.
- `--tokenizer-dir DIR` defaults to ROOT/tokenizer for production. Supply it
  explicitly for a standalone window. The consumer checks the aggregate frozen
  tokenizer identity, manifest pins, provenance and payload checksums, tensor
  layouts, vocabulary bounds, sidecars and token totals.
- Foundation loss masks must supervise exactly the real causal targets.
  `instruction`, `code_instruction` and `tool_instruction` stages refuse
  explicitly; assistant-only SFT masking is not implemented.

`data.target_tokens` (or explicit `--max-input-tokens N`) is a cumulative
real-input-ID cap, independent of `max_steps`; production requires a cap.
For `x = sequence[:-1]`, a row counts `min(valid_len, sequence_length - 1)`.
This excludes padding but includes a padded row's last real input even when its
next target is ignored. The sampler admits the longest fitting prefix of the
unchanged sample order, never skips rows to fill a remainder and never
overshoots. A remainder smaller than the next row is left unused. Metrics and
the final report distinguish cumulative input IDs from session throughput.

Checkpoint schema 2 stores `resume_state.input_tokens_seen`. Resume binds
ordered train/validation membership, shard/length/mask hashes, window and
preparation-manifest identities, tokenizer, model/training/data/run config,
stage and token cap. It also checks the real-token counter against the saved
cursor. Older checkpoints require an explicitly audited schema-2 migration;
no cumulative count is guessed. Optimizer, GradScaler, RNG and single-pass
cursor state remain preserved. Production and token-budget paths require one
GPU; the legacy DDP trial is not a production resume path.

The dataset audits shards sequentially, keeps compact per-row length metadata,
and lazily caches one token shard per worker, not the full corpus. The existing
global permutation can reload shards frequently; full-scale I/O throughput and
memory remain unmeasured. No model architecture, hyperparameters, tokenizer or
preparation recipe changed.

### Explicit incremental-dataset resume

This mode is an audited handoff, not a fingerprint override or `--init_from`.
Only fully consumed **foundation** segments can be extended. The new production
root must contain only the next contiguous owned windows (e.g. 240+); it need not
contain the old shards. No preparation or training is authorized by this feature.

Proposed invocation, **not a launch instruction**:

```bash
python -m training.train --config extension.yaml --resume --extend-dataset \
  --parent-fingerprint PARENT_FINGERPRINT \
  --parent-production-root /prepared/windows-21-239 \
  --new-data-manifest /prepared/windows-240-plus/preparation-manifest.json \
  --validation-root /prepared/windows-21-239 --extension-seed 240
```

- Preserve the verified parent checkpoint; stage a copy as `latest_checkpoint.pt`
  in an isolated output directory configured by `checkpointing.output_dir`.
  Retain the parent's exact `model`, `training`, `data`, `run` sections and
  cumulative token cap. Other configs, including the default fresh config with
  different validation settings, cannot substitute for the fingerprint-bound one.
- The original schema-2 checkpoint contains only a fingerprint/counters, **not**
  shard records or a schedule snapshot. Supply the complete, hash-verified parent
  preparation output (including `state.sqlite3`, manifests, shards and sidecars);
  for step 165 this is the windows-21–239 backup, not an invented prefix of the
  new output. The loader reconstructs and checks its fingerprint, full cursor and
  real-ID total. Missing RNG/scaler history and legacy cursor migrations refuse.
- The new preparation must directly pin that parent manifest and retain its
  recipe, pipeline and aggregate tokenizer identity. Closed, hash-verified SQLite
  indices must retain all old document identities; each new provenance document
  must be unique, absent from the parent index and present in the new index.
  This includes validation and earlier preparation history, without loading the
  global document set into RAM. Producer admission/packing remains trusted;
  these checks are normalized exact deduplication, not exhaustive near-dedup.
- Validation remains the **original fixed set**, in original order. New validation
  stays separate and unused, and cannot later enter training. Pass that same
  `--validation-root` on every extension and ordinary production resume.
- Schema stays 2. Metadata records `parent_fingerprint`, `new_fingerprint`,
  `lineage_chain`, consumed-segment/index identities, a config/schedule context
  snapshot and `extension-permutation-v1` with its explicit seed. `sample_cursor`
  and `input_tokens_seen` remain cumulative; the new segment starts locally at
  zero. The token cap remains cumulative and unchanged: whole rows only, no
  skipping/truncation or overshoot, and no extension after exhausting the cap.
- Activation atomically replaces only the staged latest checkpoint's metadata
  before any new update. Model, AdamW, scaler, schedule step/applied counters and
  RNG are retained; dedicated worker seeds do not advance the restored model RNG.
  A crash leaves either the original parent or exactly one activated handoff.
  After activation, **remove extension/parent/new-manifest/seed flags** and use
  `--resume --production-root NEW_ROOT --validation-root ORIGINAL_ROOT`.
  Repeating the old handoff refuses; a subsequent extension requires consuming
  the current segment completely and explicitly supplying its fingerprint/root.

For the verified step-165 parent, the original `smoke.yaml` SHA is
`f009b67a8b0d302b7f487e0c2d73a4be4d553f295c92564e14d68ee7a410793a`.
It binds max LR `0.0006`, warmup 500 and max steps 16,221. The checkpoint stores
LR `0.000198`, but no scheduler/config snapshot; reconstructing its fingerprint
with that config establishes update 166's LR as `0.0001992`, not a warmup reset.
The extension CPU regressions are in `tests/test_dataset_extension.py`; no GPU
execution or real new-window handoff has been verified by these synthetic tests.

Extension verification on Python 3.14.2 / PyTorch 2.14.1+cpu: **126 passed,
9 skipped, 120 subtests passed** across the full local suite. The skips are eight
Kaggle-only preparation integrations and one optional Kaggle-SDK test. The actual
trainer was exercised on tiny CPU fixtures with its hardware gate/autocast
overridden only in the test harness; production remains GPU-gated.

Historical consumer-only CPU verification: 93 tests passed, 1 optional Kaggle-SDK test
skipped, 93 subtests passed. The runnable subset excluded
`test_data_quality.py`, `test_fresh_preparation.py` and
`test_production_preparation.py` because their imports need unavailable
`datasketch`/`tokenizers`/`transformers`. A separate manifest-to-DataLoader smoke
counted 10 real IDs in 14 input slots under a strict 10-ID cap. This is not
GPU, full-corpus, training-quality or cloud-readiness evidence.

### Validation trend watchdog

`val_trend_diverged` stops immediately on NaN or either infinity. For finite
losses it compares the latest `half`-evaluation mean with the best earlier
complete, non-overlapping block mean, stopping on a rise **above 0.05**.
Using the best earlier block also catches slow cumulative reversals that an
adjacent-window difference would miss forever.

The trainer uses `half = max(2, 5 // val_steps)`: every-update validation uses
five-point means (ten-evaluation startup); validation every 100 updates uses
two-point means (four-evaluation startup), avoiding a new 1,000-update blind
period. These are heuristics, not statistically calibrated confidence bounds.
The real step-166–185 series passes every prefix with either window size.

History is process-local, not checkpointed: resumes restart this finite-trend
startup period and lose the previous best mean. Short repeated sessions can
therefore evade the trend guard. Sparse validation provides no wall-clock
guarantee; keep the approved launcher's independent deadline and hard nonfinite
gradient/metric/checkpoint checks. The trend check does not replace those guards.
No validation membership, config values, identity inputs or checkpoint format
changes are needed for this policy.

## Stage A benchmark results

Measured on the checkpoint above, using greedy pass@1:

| Benchmark | Result |
|---|---:|
| HumanEval (164 problems) | 7/164 (4.27%) |
| MBPP full test set (500 problems) | 13/500 (2.60%) |

These are measured results, not expected targets. Important protocol limits:

- The first naive plain-prompt run scored 0/164 and 0/500. The reported run used inference-only changes: a document-start EOS prefix, non-EOS special-token masking for plain completion, and removal of the trailing prompt newline.
- MBPP used a description and three tests in a docstring followed by a named `def` prefix. The stop rules can cut off helper functions.
- Prompt/decoding choices were tuned by inspecting the first 40 problems of each benchmark, so the scores may be optimistic. This was not weight training on the benchmark problems, but it is not an untouched evaluation protocol either.
- The decoder was checked against the model forward pass (maximum logit difference about 4e-6). Reference solutions passed executor checks on 40 HumanEval and 60 MBPP problems, not the entire reference suite.

The committed harness and wrappers record the protocol. Both wrappers run the combined evaluation and select the reported MBPP style:

```bash
python evaluation/humaneval_runner.py
# or: python evaluation/mbpp_runner.py
```

The harness expects access to the private checkpoint through `HF_TOKEN` or a Kaggle secret named `HF_TOKEN`. Keep tokens in a secret store or the environment, never in source files. It installs missing evaluation dependencies and uploads results to the checkpoint repository by default; set `CF_NOUPLOAD=1` to disable uploads. Model architecture is fetched from GitHub master, so pin that code as well when reproducing results.

## Progress update - October 2, 2026

- **Stage A2 completed:** 150-step weights-only continuation on code-only FIM data. HumanEval remained **7/164 (4.27%)** and MBPP **13/500 (2.60%)**, under the same previously tuned greedy protocol. No benchmark improvement is claimed.
- **Data-format fix:** FIM is now applied only to StarCoder/Python and CodeParrot code; prose, instruction and tool-call data are not FIM-transformed. CommitPackFT is not double-wrapped.
- **A3 preparation:** the corrected build completed with 61 shards, 60,459 sequences (123,820,032 tokens), 18,989 retained Evol documents and 8,609 retained Glaive documents. All 61 tensors and the manifest were independently read back from the private `stageA3-v2/` folder. Evol/Glaive examples may repeat earlier training; code-stream offsets are estimates, not proof of disjoint data.
- **Training safeguards prepared:** isolated checkpoint uploads with remote size readback, periodic saves and an absolute wall-clock deadline. The code/config and tests are published. These safeguards do not guarantee a better benchmark score.
- **Stage A3 result:** A3 stopped cleanly at step 38 after 19,913,216 training tokens in a one-hour-cap chunk. The isolated step-38 checkpoint was uploaded/read back and evaluated: HumanEval 7/164 (4.27%, unchanged), MBPP 14/500 (2.80%, one extra problem versus A2). Total job runtime including evaluation was 76m25s. This small change does not establish broad improvement. Stage A2 stays untouched; A3 is not promoted.
- **Tool calling remains a goal:** assistant/tool-token-focused SFT using Vision action and nexus-agent tool schemas, plus separate tool-selection, argument and execution tests. The serving bridge remains a stub.

## Tool-calling experiments - October 4-5, 2026

Small tool-use SFT runs started from the A2 checkpoint, using Vision action and nexus-agent tool data mixed with plain code data. A2 stays the reference checkpoint; no experiment below is promoted.

| Run | Right action | Exact args | No-call correct | Missed calls | HumanEval | MBPP |
|---|---:|---:|---:|---:|---:|---:|
| A2 (reference) | - | - | - | - | 7/164 | 13/500 |
| T2 | 255/403 | 186/403 | 22/38 | 35 | 3/164 | 4/500 |
| T3 | 267/403 | 182/403 | 2/38 | 7 | 5/164 | 12/500 |
| T4 | 203/403 | 176/403 | 37/38 | 178 | 5/164 | 11/500 |

What it shows: tool-calling is a balancing act. T3 kept most coding skill but called tools when it should not (no-call correct 2/38). T4 fixed that (37/38) but skipped tools on many real requests (178 missed calls). Fixing one failure produced the other. Next step is to balance the no-call data.

Caveats: single seed per run, a synthetic Vision evaluation set, small test sizes, and the same tuned greedy HumanEval/MBPP protocol as above. Treat differences of a few problems as noise.

## Architecture

The 250M configuration uses:

- 16 decoder layers, hidden size 1,024, intermediate size 2,816.
- 16 query attention heads and 4 KV heads (GQA).
- RoPE, RMSNorm, SwiGLU and PyTorch scaled dot-product attention.
- A custom 32,000-token tokenizer and 2,048-token context.
- Separate FIM, tool-call, tool-result, thinking and JSON special tokens. Having these tokens does not prove reliable tool use.

See [`configs/config_250M.yaml`](configs/config_250M.yaml) and [`models/architecture.py`](models/architecture.py). The repository also contains larger configuration files; they are not trained or evaluated releases. SDPA does not guarantee a particular FlashAttention backend, memory footprint or freedom from OOM. This README makes no validated edge-device latency claim.

## Stage A data and its limits

The configured source weights were:

| Source | Configured weight |
|---|---:|
| `bigcode/starcoderdata` (Python) | 56% |
| `codeparrot/codeparrot-clean` | 22% |
| `HuggingFaceFW/fineweb-edu` | 13.5% |
| `theblackcat102/evol-codealpaca-v1` | 5.3% |
| `glaiveai/glaive-function-calling-v2` | 2.5% |
| `bigcode/commitpackft` (Python) | 0.7% |

These are sampling targets, not measured post-filter token shares. Per-source post-dedup counts were not retained, so exact retained token allocations cannot be claimed.

The legacy Stage A corpus predates the current preparation fixes. Correcting
the code does not retroactively correct its data:

- `data/deduplicate.py` now uses seeded datasketch MinHash/LSH and verifies candidate Jaccard similarity, alongside exact normalized SHA-256 matching. The old corpus used ordered-shingle hashes. Neither establishes a globally unique-token corpus.
- `data/filter_quality.py` uses Python AST checks for pure Python and preserves provenance. The old character-bracket check has been removed; regression tests cover brackets inside valid strings/comments.
- `data/tokenize_dataset.py` applies 50% FIM only to approved code sources, never prose/instruction/tool data or preformatted FIM. It preserves line endings and records padded-tail valid lengths.
- No completed decontamination audit of legacy Stage A is claimed. Fresh preparation records the scope and limits of its explicit lexical exclusion.

The nominal batch is 8 sequences × 32 accumulation steps × 2,048 tokens = 524,288 tokens per schedule step. Actual valid-row handling, partial batches and skipped optimizer updates mean this product is not proof of the exact number of useful training tokens consumed.

## Training and legacy checkpoint caveats

Stage A used free Kaggle GPU sessions on **one account**, with checkpoints uploaded between sessions. Multi-account quota rotation is not the operating plan. More pretraining depends on available compute; no paid-compute commitment is implied by this README.

The Windows workspace supervisor (`../.watch/rotation.py`) is restricted to
account A, `yashbajpai2027`. Other accounts cannot be selected or launched,
including when old ledger entries still mention them. Its owner kernel reads
`HF_TOKEN` from Kaggle Secrets; enable that secret for the notebook before
running it. Never insert credentials into kernel source. The
`../.watch/STOP_WATCHDOG` marker keeps automatic training stopped during setup.

The audited resume path fixes short-shard handling and separates the schedule counter from optimizer-applied updates. It stores sample cursor, data/tokenizer fingerprints, GradScaler and RNG state. These fixes prevent the known errors going forward; they cannot undo or prove harmless the invalid/uninitialized rows or counter errors consumed by an older checkpoint.

The former step-2,100 pause and loss table were historical snapshots, not the current state. Training loss is not code accuracy, and no final loss or throughput claim is substituted for the benchmark results above.

For safe resume details, see [`docs/stage_a_resume.md`](docs/stage_a_resume.md). After explicitly staging and verifying the intended checkpoint, tokenizer and shards:

```bash
git clone https://github.com/Yash1bajpai/CodeForge-250M.git
cd CodeForge-250M
pip install -r requirements.txt
python -m unittest discover -s tests -v
python training/train.py --resume --max_hours 6
```

Do not use the example to resume completed Stage A blindly. Legacy checkpoints require the audited migration record described in the resume guide. The audited path supports one GPU. Stop/save limits apply at batch boundaries; checkpoint serialization and uploads need extra time.

## Checkpoints

The [public Hugging Face repository](https://huggingface.co/Yash1bajpai/CodeForge-250M) contains an older checkpoint. It must not be presented as the measured Stage A revision or as a ready-to-use nexus-agent integration. The evaluated revision is identified above; the public checkpoint has not been promoted to that result.

### Phone-first control plane and cloud backups

The verified control plane runs on Android Termux with a Debian environment for
the coding agent and cloud clients. `scripts/phone_shell.sh` is the native Termux
`codeforge` launcher; `scripts/setup_phone.sh` installs the Debian controller.
The shared repository path is `~/projects/own_llm/CodeForge-250M_win`, external
controller state is `~/projects/own_llm/.watch`, and small staging/evidence lives
under `~/codeforge-storage`. Large datasets/checkpoints and all ML execution remain
cloud-side. Do not install the root ML `requirements.txt` on the phone.

See **[the migration checklist and handoff](docs/phone-migration.md)** for key-only
USB SSH, private credential paths, tmux/wake-lock risks, evidence and exact backup
commands. Installation, provider access and Drive upload/readback passed on
October 26, 2026. Windows is not required for normal phone-side control.
Legacy Windows rotation/watchdog files are preserved references, not phone
startup services.

From native Termux's normal `$` prompt, run one short command:

| Command | Purpose |
|---|---|
| `omp` | Open the project using the verified Antigravity login |
| `cf-check` | Live provider/cloud verification with PASS/FAIL/ACTION results and an automatic private report |

No manual Debian activation or long model selector is needed for these commands.
The native installer is `scripts/install_phone_commands.sh`; supplied OMP
arguments and subcommands still work normally. Codex remains blocked by any
outstanding private revocation/re-login security gate; the checker does not hide it.

These do not authorize training or another CPU preparation. Never place
`CF_TRAINING_APPROVED=1` in phone startup files; only a separately approved cloud
training invocation may receive it.

`scripts/kaggle_drive_backup.py` downloads saved outputs from **only**
`yashbajpai2027` into a fresh **local** staging folder. It does not itself upload
to Drive, mount Drive in Kaggle, start training, or grant Kaggle Google access.
It uses the owner's `~/.kaggle/access_token`, checks the authenticated username,
requires a completed notebook/current saved version and refuses existing
destinations. The phone handoff documents `rclone copy` (never `sync`) followed
by explicit cloud readback/check evidence. A dedicated personal OAuth client is
configured; it does not rely on rclone's quota-limited shared Google client.

The historical Windows bridge used Drive for desktop, requiring the PC/network/
desktop sync process to stay running. Its example is retained for reference,
not as a phone dependency:

```powershell
python scripts/kaggle_drive_backup.py --kernel yashbajpai2027/codeforge-drive-storage-check --version 1 --destination "G:\My Drive\codeforge-storage\logs\kaggle-storage-check-v1"
```

That private notebook is a CPU-only storage check with GPU, TPU, internet and
external data disabled. Its source and metadata are
`kaggle/drive_storage_check.py` and `kaggle/drive_storage_check-metadata.json`.
Do not push the whole `kaggle/` directory to repeat it: that directory's default
`kernel-metadata.json` belongs to the training notebook.

For a training/data-prep output backup, substitute its owner notebook slug,
current version and a new folder under `checkpoints/` or `datasets/`. Downloads
stream in bounded chunks, check HTTP lengths, and verify SHA-256 by local
readback. A final `backup-manifest.json` records the files without signed URLs.
Partial failures never produce a completion manifest. The storage-check artifact
also has a producer-side checksum in `storage-check-manifest.json`.

This command is on demand, not a scheduled backup or a live checkpoint uploader.
Only files saved into Kaggle notebook outputs are available; it cannot recover
unsaved files from a killed session. Keep the existing checkpoint persistence.
Drive cloud completion is separate: check the resulting file on the Drive website.
Neither a local write nor `backup-manifest.json` claims cloud verification.

## Next work

1. Evaluate corrected plain/FIM behavior on a fresh checkpoint under a fixed protocol; preparation alone does not prove generation quality.
2. Build tool-use SFT from Vision's action registry and nexus-agent schemas, with no FIM on chat/tool data and loss focused on assistant/tool tokens. Evaluate tool selection, argument validity and execution outcomes separately from code benchmarks.
3. Replace the serving placeholder with real inference, then test any JSON constraints and integrations rather than claiming guaranteed tool calls.
4. Scale cross-batch deduplication, bounded-memory loading and contamination checks before increasing the pilot corpus toward the production budget.
5. Validate source access, licensing, retained counts, compute and evaluation gates. Freeze the production tokenizer before model training; the candidate CPU pilot does not promote it automatically.

Reaching 10B tokens is a training milestone, not a promise of quality or the end of development.
