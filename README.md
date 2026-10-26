# CodeForge-250M

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-SDPA-ee4c2c)](https://pytorch.org/)

CodeForge-250M is a roughly 246M-parameter decoder-only code model trained from scratch. The goal is to support code completion, FIM editing and tool calling for nexus-agent and Vision. Those are development goals, not validated capabilities of the current checkpoint.

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
schedule is 16,221 nominal updates at 256 sequences of 2,048 tokens; exact valid
token accounting, instruction loss masking and the production data-window
schedule must be checked before larger training. Existing checkpoints are not
initialization sources and are not overwritten.

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

For a separately authorized future private, owner-only **CPU** build, use a new report filename. **Do not run this during phone migration: another preparation launch is not authorized.**

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

After phone setup, the read-only control-plane check and explicit Drive storage
probe are:

```sh
python scripts/phone_control.py status
python scripts/phone_control.py drive-probe --remote codeforge-drive:codeforge-storage
```

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
