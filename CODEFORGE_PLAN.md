# CodeForge — full plan and continuation handoff

Snapshot: October 26, 2026. This file consolidates the project decisions, execution plan, known evidence, remaining gates and another-PC handoff. Live job status must be rechecked; this document is not a live dashboard.

## 1. Non-negotiable operating decisions

- **The PC is a controller only.** Edit/sync code and configurations, submit/monitor Kaggle notebooks, inspect small reports and coordinate backups from the PC.
- **All substantive workloads run on Kaggle:** corpus download/processing, filtering, deduplication, tokenizer fitting, tokenization/packing, model training and evaluation. Do not move these onto the PC or phone.
- Only Kaggle owner **`yashbajpai2027`** is enabled. Other saved account profiles are inactive; do not enable rotation or submit work using them.
- Kaggle CPU data preparation is authorized. **GPU pilots and model training are NOT authorized. Notify Yash and obtain explicit approval before launching either.** Never set `CF_TRAINING_APPROVED=1` in startup files, persistent environments or ordinary setup commands.
- Yash handles account selection, logins, OAuth consent, passwords and verification codes. Authentication work is deferred; do not resume it unprompted.
- The phone installation is preserved but is not the active workflow. Do not reopen phone migration/setup work.
- Preserve all legacy checkpoints and historical data. The fresh run starts from random weights, not from a legacy checkpoint.
- Do not fabricate unique-data counts, completed uploads, test results, executed commands or model capabilities. RUNNING is not proof of healthy processing.
- Keep credentials outside source, Git, notebook code, plan files and chat. Use private account stores/Kaggle Secrets. Never print authorization exports or copy whole credential directories into a handoff.

## 2. Goal and definition of success

Build a roughly **246M-parameter code model from scratch**, targeting **10B non-padding training-input tokens** by **January 1, 2027**.

Desired capabilities:

1. Generate and edit code, including fill-in-the-middle (FIM).
2. Explain written code and follow coding instructions.
3. Select tools and emit valid names/arguments; correctly choose not to call a tool when unnecessary.
4. Participate in an agent loop that writes files, runs commands/tests, reads actual results and fixes failures.
5. Integrate with **Nexus** and **Vision** using their real tool/action schemas.

The model does not execute code by itself. An agent/runtime must parse its requests, execute allowed tools in a controlled environment and return real results. Public function-call conversations alone do not establish reliable coding-agent behavior. Ten billion tokens is a training milestone, not a guarantee of matching larger models.

Completion requires measured coding/FIM/tool-use behavior, real inference and end-to-end integration—not merely a checkpoint or a token count.

## 3. Current state: completed versus unverified

| Area | Evidence/state |
|---|---|
| Fresh model configuration | `configs/config_250M_fresh.yaml`; random initialization; explicit training-approval gate |
| Legacy model | Stage A completed at step 2,898, approximately 1.52B intended input tokens; later experiments preserved, not initialization for this run |
| Legacy coding quality | Reference Stage A/A2 measured HumanEval 7/164 and MBPP 13/500 under a previously tuned greedy protocol; not an untouched evaluation baseline |
| Fresh CPU data pilot | `yashbajpai2027/codeforge-fresh-cpu-preparation-139cbd2a`, version 1 COMPLETE |
| Pilot retained data | 3,272 train documents / 6,787,976 stored real token IDs; 61 validation documents / 182,325 IDs |
| Pilot tokenizer | 32K candidate; does not establish production promotion |
| Pilot backup | 24 producer files, 45,759,067 bytes; previously checksum-verified and independently compared against Drive downloads, zero differences |
| Production recipe | `configs/fresh_data_production.yaml`, pinned revisions/license allowlists; corrected first window and real resume verified, larger-window preparation now underway |
| Initial preparation notebook | `yashbajpai2027/codeforge-fresh-cpu-prep-534125d9/1` ended **ERROR**: prepared its bounded window, then the Python subprocess failed to exit and hit its 15,500-second timeout |
| Published original artifacts | Independently audited on Kaggle: frozen 32K tokenizer, matching sample/config/DB hashes and 10 windows containing **5,045,576 non-padding input IDs** plus 2,446,008 padding IDs. Old state/shards remain **unpromoted**, not an addition to fresh training-token totals |
| Preparation contracts | Corrected `codeforge-recovered-cpu-prep-2d372b23/1`: **76 tests passed in 14.882s**; real-resume `d4d37022/1`: 76 passed in 15.142s. Both actual CLI subprocesses exited normally |
| Live source preflight | `codeforge-fresh-cpu-prep-ca3166c4` v1 COMPLETE: all seven pinned sources produced an admitted, normalized example; 71 tests also passed in 20.180s. This does not measure full-corpus supply or quality |
| Real separate-notebook resume | `codeforge-recovered-cpu-prep-2d372b23/1` → `codeforge-fresh-cpu-prep-d4d37022/1`, both COMPLETE; windows 10 → 20, same tokenizer, advancing cursors and persistent global dedup; independent audit found 8,330 distinct normalized documents with no repeated document across the chain |
| Current prepared data | **10,065,678 train input IDs + 101,480 validation IDs**; 4,445,322 padding IDs counted separately; 20 audited windows. These are prepared data, not consumed training tokens. Original legacy windows are excluded |
| Verified cloud backup | Private dataset `yashbajpai2027/codeforge-fresh-prepared-w20-20261026/1`; separate notebook `codeforge-cloud-backup-check-80f08122/1` read back every file: **184 producer files / 195,085,969 bytes**, exact length/SHA-256 matches |
| Shutdown repair | Bounded diagnostic reproduced surviving native HF range I/O after main returned. Parquet prebuffering disabled; iterators/SQLite/session resources explicitly closed. Fixed producer and real-resume jobs both completed normally, without suppressing exceptions or forcing process exit |
| Current active preparation | `yashbajpai2027/codeforge-fresh-cpu-prep-fe52fd65/1`: latest observed RUNNING, resuming `d4d37022/1` with a **100M additional input-token operational cap**. Its new token counts are not yet verified |
| Local execution guard | Exercised on the PC; refused before creating a preparation output directory |
| Fresh model training | **Not started** |
| Original output preservation | 93 files / 139,316,294 bytes downloaded with HTTP-length/SHA-256 readback; mounted Drive copy readback also passed. A private Kaggle recovery dataset is ready and its candidate/windows were audited from a separate notebook. Independent Drive sync and HF backup remain unverified |
| Kaggle HF checkpoint access | `HF_TOKEN` secret-binding/readback gate remains unresolved; working controller HF access is not proof of notebook checkpoint access |
| Serving | `serving/nexus_bridge.py` remains a placeholder, not working model inference |

The two superseded submissions `codeforge-fresh-cpu-prep-33b6d9c8` and `codeforge-fresh-cpu-prep-64a7cc29` were deleted after the source-schema correction. Do not resume or use them.

Active CPU preparation notebook:
https://www.kaggle.com/code/yashbajpai2027/codeforge-fresh-cpu-prep-fe52fd65

Latest completed real-resume checkpoint: `yashbajpai2027/codeforge-fresh-cpu-prep-d4d37022/1`.

Original production attempt, preserved after its shutdown failure:
https://www.kaggle.com/code/yashbajpai2027/codeforge-fresh-cpu-prep-534125d9

Current Windows project:
`C:\Yash\own_llm\CodeForge-250M_win`

Controller evidence outside Git:

- `../.watch/fresh-production-launch-20261026-02.json`: exact notebook version, configuration/source hashes and operational limit.
- `../.watch/production-observed.json`: controller observation; check freshness before treating it as current.
- `../.watch/observe-production.py`: original PC observer, exited after recording ERROR; it did not restart or resubmit the failed job.
- `../.watch/production-error-backup-534125d9/`: all 93 saved producer artifacts plus the verified backup manifest; no tensors were loaded or data prepared on the PC.
- `../.watch/fresh-production-regression-before-20261026-01.json` and `production-regression-before-observed.json`: six behavioral regressions observed failing on Kaggle before their fixes.
- `../.watch/fresh-production-regression-after-20261026-01.json`, `production-regression-after-observed.json`, and `production-regression-verified-5eb73756/`: 71-test pass, downloaded verification/source receipts and exact submission identities.
- `../.watch/fresh-source-verification-20261026-01.json`, `fresh-source-verification-observed.json`, and `source-verification-reports-ca3166c4/`: current-code live preflight for all seven sources. Kaggle `HF_TOKEN` remained unavailable with `backend-request-failed`; this gate is still open.
- `../.watch/cloud-fixture-{seed,resume}-launch.json` and corresponding `*-observed.json`: separate CPU producer/consumer notebook proof, not production-data counts.
- `../.watch/cloud-fixture-seed-backup-655fde13/` and `cloud-fixture-resume-backup-1dd18381/`: verified complete fixture output chain.
- Drive fixture copies: `G:\My Drive\codeforge-storage\datasets\cloud-resume-fixture-proof-20261026-655fde13-1dd18381`. Mounted-path readback is not independent cloud-sync proof.
- Private recovery dataset: `yashbajpai2027/codeforge-prep-recovery-534125d9-20261026/1`; privacy verified through the owner API, source file layout retained.
- `../.watch/production-recovery-audit-reports-6d2d3672/` and `recovery-shutdown-before-observed.json`: separate Kaggle artifact audit and reproduced shutdown failure with native HF range-read stacks after main returned.
- `../.watch/fresh-recovered-production-launch.json` and `recovery-shutdown-after-observed.json`: exact corrected source snapshot, preserved tokenizer, 76-test pass and real preparation CLI normal exit under the bounded 300-second smoke.
- Preserved original Drive copy: `G:\My Drive\codeforge-storage\datasets\production-error-preserved-534125d9-20261026`; mounted readback passed, independent cloud sync unverified.
- `../.watch/fresh-real-resume-launch.json`, `real-resume-observed.json`, and `real-resume-audit-reports-3b01d679/`: actual v2 resume, complete 20-window audit, byte/control round trips and observed held-out compression for 17 source/language groups.
- `../.watch/production-recovered-backup-2d372b23/` and `production-real-resume-backup-d4d37022/`: complete verified producer backups; Drive copies are under `datasets\fresh-production-v2-windows-000001-000020-20261026`.
- `../.watch/prepared-chain-dataset-published.json`, `cloud-backup-readback-observed.json`, and `cloud-backup-readback-reports-80f08122/`: independently read-back private Kaggle backup, not a claim of Drive cloud sync.
- `../.watch/fresh-scaled-preparation-launch.json` and `scaled-preparation-observed.json`: current 100M-cap CPU job. `../.watch/observe-cpu-prep.py` runs as `CodeForgeCPUScaleObserver`; it records sanitized status/logs and does not launch another job automatically.
- Tested source/handoff copy: `G:\My Drive\codeforge-storage\source\fresh-prep-tested-source-20261026-w20` contains the exact 44-file tested source/config ZIP, this plan, README/AGENTS, selected credential-free receipts and the observer script. This is not a full Git workspace or chat/session export; authentication was not copied. Restore the observer under sibling `.watch/` to preserve its relative project layout. Mounted readback is verified, independent Drive sync is not.

The current PC observer is a local process. It does not migrate to another PC with the conversation. Kaggle execution does not require the PC to remain on, but local monitoring stops if its host stops.

## 4. Model and training configuration

| Setting | Fresh plan |
|---|---:|
| Parameters | 245,924,864 |
| Decoder layers | 16 |
| Hidden size | 1,024 |
| Intermediate size | 2,816 |
| Query / KV heads | 16 / 4, GQA |
| Context length | 2,048 |
| Vocabulary | 32,000 |
| Initialization seed | 42 |
| Nominal total batch | 256 sequences |
| Foundation learning rate | 0.0006, minimum 0.00006 |
| Warmup | 500 steps |
| Foundation nominal update cap | 16,221; actual shifted valid-token accounting must be verified |

Architecture uses RoPE, RMSNorm, SwiGLU and PyTorch SDPA. These choices do not prove a particular attention backend, throughput or memory fit.

Precision, accelerator support, batch size and wall-clock feasibility must be measured on the actual authorized Kaggle hardware. Do not assume BF16 is suitable for every available GPU.

## 5. Dataset recipe and honest budget accounting

| Source/subset | Stage | Training-input allocation | Pinned revision |
|---|---|---:|---|
| `codeparrot/github-code` | Foundation | 5,950,000,000 | `b5661e6b17396364b2bcf8e68977b0d28e1ebd19` |
| `codeparrot/codeparrot-clean-train` | Foundation | 1,275,000,000 | `3e6ab65f2864931e041f6a82db9b5a6ec2b71ab4` |
| `HuggingFaceFW/fineweb-edu`, sample/10BT | Foundation | 1,275,000,000 | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| OpenCoder stage1, largescale_diverse_instruct | Code instruction | 800,000,000 | `1bcab575f5e2d1c1fd6652720418524c27b3d58b` |
| OpenCoder stage1, realuser_instruct | Code instruction | 200,000,000 | Same OpenCoder revision |
| `glaiveai/glaive-function-calling-v2` | Tool instruction | 300,000,000 | `e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac` |
| `NousResearch/hermes-function-calling-v1`, func-calling files | Tool instruction | 200,000,000 | `dae3e1d28cfbcf4b915c04ea1e072030529b4bda` |

Stage totals: **8.5B foundation + 1B code instruction + 0.5B tool instruction = 10B planned training-input tokens**.

These are allocations, **not assertions of available unique tokens**. In particular, Glaive/Hermes source JSON totals only about 320MB; that does not establish 500M unique tool tokens. Prepare admitted records once, report actual retained counts and source exhaustion, and do not silently repeat data to fill a shortfall. Any later multi-epoch/rebalanced curriculum must document the repetition and be decided before training.

### Source and license policy

- Public, ungated, immutable source revisions; built-in JSON/Parquet readers only. Do not execute remote dataset scripts.
- GitHub code: approved per-record repository-license labels only—MIT, Apache-2.0, BSD-2/3-Clause, ISC, CC0 or Unlicense as configured. Clean Python uses its narrower YAML allowlist. Reject unknown/unapproved labels.
- FineWeb-Edu: declared ODC-BY; OpenCoder: declared MIT; Glaive/Hermes: declared Apache-2.0.
- Retain source repository/revision/file/row, original repository/path/URL when available, license label and content identity.
- Dataset labels are not an independent per-file licensing/privacy audit. Keep attribution and document known limitations.
- Python AST checks apply to Python, not JavaScript, TypeScript, shell, Markdown or all source code indiscriminately.

**Important schema correction:** raw GitHub-code Parquet has `repo_name`, `path`, `size`, `content`, `license`. Its old loader synthesized `code` and `language`. Read `content`, derive language from the path and reject missing required fields rather than scanning the entire source while rejecting every row.

## 6. Tokenizer, quality, split and packing rules

- Fit one **32K byte-level BPE** tokenizer on a deterministic train-only sample: 56M characters multilingual code, 12M clean Python, 12M FineWeb-Edu; approximately 80M characters total, with bounded whole-document overshoot.
- Preserve the existing 11 distinct EOS/unknown/padding/FIM/tool/thinking/JSON special tokens and full byte alphabet.
- Validate vocabulary size, special-token IDs, round trips, unknown-token behavior and held-out/per-language compression before promotion.
- Freeze the tokenizer identity for the data chain. Do not refit it during each resumed window or after model training begins.
- Hold out approximately 1% validation using repository grouping when repository identity exists, otherwise normalized content identity. Do not fit on validation. Reject validation matches to tokenizer-sample exact hashes or sampled repeated blocks.
- Use global SQLite-backed normalized-exact SHA-256 deduplication plus up to eight sampled repeated 256-word blocks per long document. This conservative filter does not establish exhaustive near-deduplication or universal train/validation independence.
- Exclude matching high-specificity 32-lexical-token windows from pinned HumanEval/MBPP prompts, solutions and tests. Full benchmark decontamination is not established.
- Deterministic 50% FIM only for eligible foundation code. No FIM on prose/Markdown or instruction/tool conversations.
- Normalize instructions and tool traces into consistent system/user/assistant/tool turns. Preserve real tool arguments/results. Never execute downloaded examples during normalization.
- Supervise assistant contents/EOS for SFT, not system/user/tool-result text or padding. Reject over-context conversations with counts rather than silently truncating tool traces.
- Pack `uint16` IDs into 2,048-token rows; retain valid-length and assistant-loss-mask sidecars, checksums and per-source provenance.
- Track source input IDs, stored valid IDs, padding, discarded tails, supervised targets, documents and rejections separately. They are not interchangeable training-token counts.

## 7. Kaggle windowing, persistence and restart plan

Kaggle maximum session durations are 12 hours for CPU/GPU and 9 hours for TPU—not a nine-hour waiting gap between sessions. The current work is CPU-only, not TPU.

Production preparation defaults:

- Four-hour engine deadline, with clean shutdown margin.
- Roughly 100M encoded input tokens per session, or earlier stop at the disk-space floor.
- First corrected session uses a reduced **5M-input-token cap**; this is a bounded validation window, not a reduction of the overall target.
- Checkpoint after bounded source-row batches; immutable data windows plus a single SQLite transaction for state, cursor/dedup indexes and checkpoint metadata.
- Hash and independently read back each output window before treating it as committed.
- Resume only from the exact completed saved owner notebook version, with matching recipe, checkpoint, preparation-code and Python/package identities; pin the producer's Kaggle runtime image.
- Carry the global duplicate index, source file/row cursors and counters forward. Do not restart at row zero and recount data.
- Earlier data windows remain in parent notebook outputs. Keep the entire parent chain; copying the newest checkpoint alone is not a complete dataset backup.

The observer does **not** automatically launch more notebooks. Do not blindly resubmit while a run is active, or interpret the absence of logs as proof of progress.

Checkpoint format **v2** recovers committed state when the external receipt is missing/stale, rolls back copied SQLite hot journals, and preserves the last committed window after an exception before rethrowing the original failure. An interrupted tokenizer publication can recover its receipt without refitting or changing the tokenizer. Regression checks exercise these failure boundaries.

Kaggle explicitly rejected ERROR notebooks as inputs in the CPU attachment smoke; completed `owner/slug/version` inputs were mounted and read successfully. The launcher refuses ERROR/active parents before submission. Saved failure artifacts need an explicit verified recovery/publication step, not a success flag or a blind resubmission.

The failed `534125d9` producer predates v2 and the corrected tool formatter. Its original tokenizer is verified and preserved with identity `3b06dbefb2abaa646568722ef799ad20ad2c7e4ba2f195583cb29e1a52b0532e`. Explicit `--tokenizer-dataset yashbajpai2027/codeforge-prep-recovery-534125d9-20261026/1` adoption verifies recipe/sample/tokenizer and imports **only** the frozen tokenizer and training sample into fresh v2 state. It does not import the old SQLite state, cursors, windows or token totals. Re-tokenized old records count only in the newly validated output, never as both old and new tokens. No legacy-format compatibility shim or new tokenizer fit is used.

Saved Kaggle outputs are the immediate persistence boundary. Drive/HF upload is separate. Use non-destructive copies into new destinations and explicit remote readback/check evidence. Do not use `rclone sync` to mirror/delete existing backups.

Persistent destinations:

- Drive: `codeforge-storage` in the owner's 5TB Drive.
- Windows Drive-for-desktop location: `G:\My Drive\codeforge-storage`.
- Configured cloud remote: `codeforge-drive:codeforge-storage` where that remote exists; do not assume another PC is authenticated.
- Private HF model repository: `Yash1bajpai/CodeForge-250M-rotation`, isolated fresh-run prefixes.

Do not put dataset processing on the PC to work around a cloud credential problem. Unsaved files in a killed Kaggle session are not guaranteed recoverable.

## 8. Ordered execution plan and acceptance gates

### Phase A — inspect the current preparation run

1. Check the exact current notebook/version above before submitting anything else.
2. Obtain contract-test results, source-format preflight and preparation logs/outputs.
3. If the run fails, diagnose the recorded failure and correct the cause; do not blindly repeat the same submission.
4. Do not infer which internal stage has completed from RUNNING alone.

**Pass:** observed runtime results and usable producer artifacts. **Current: passed.** The original shutdown failure was reproduced and fixed; corrected producer and real-resume jobs passed 76 contracts, all-seven-source live preflight, actual process exit and independent artifact/state audits.

### Phase B — validate and freeze the production tokenizer

1. Confirm the representative sample is train-only and the source/license/split policies really ran.
2. Check 32K vocabulary, all special tokens, byte round trips and held-out compression by source/language.
3. Record tokenizer file hashes; keep the same identity for every window and all future weights.

**Pass:** independently checked tokenizer and frozen identity, not simply an existing tokenizer.json. **Current: passed for preparation.** The preserved 32K candidate passed vocabulary/special/byte-alphabet checks, four exact UTF-8/control round trips, sample/config/hash validation and observed held-out compression across 17 source/language groups. Identity remains `3b06dbefb2abaa646568722ef799ad20ad2c7e4ba2f195583cb29e1a52b0532e`; these checks do not establish model quality.

### Phase C — verify the first data window and a real resume

1. Verify source/config hashes against the submitted snapshot.
2. Read back shard IDs, shapes, lengths, masks, checksums and token accounting on Kaggle.
3. Audit representative source, explanation, no-call, call/result and multi-turn examples; verify license/provenance and rejection counts.
4. Resume a completed window in a new CPU notebook using its exact saved version. Confirm unchanged tokenizer, advancing cursors, persistent cross-session dedup and no skipped/double-counted records.
5. Verify durable backups of the entire chain before scaling windows.

**Current: passed for the bounded real-data chain.** A separate CPU notebook audited both v2 producers: exact parent receipt, source cursors, unchanged tokenizer, all 20 window checksums/layouts/masks/counts, 8,330 distinct normalized document hashes and exact agreement with the latest SQLite/source counters. All 184 producer files were also copied to a separate private Kaggle dataset and independently read back by length/SHA-256 before scaling. Drive mounted copies passed readback, but Drive cloud sync is unverified: the managed browser requires sign-in and no login was attempted.

**Pass:** real first-window and resume evidence; no token duplication concealed by new filenames.

### Phase D — scale preparation and resolve data shortfalls

1. Continue bounded CPU windows on the enabled account only.
2. Measure retained unique tokens, language/source composition, dedup/filter rates, storage footprint and preparation throughput.
3. Address insufficient instruction/tool supply explicitly. More source bytes or repeated epochs do not automatically improve quality.
4. Build domain-specific Nexus/Vision data from their actual registries/schemas when available. Include no-call decisions, missing-argument clarification, errors and recovery, not only successful calls.
5. For coding-agent execution traces, run code/tests in a controlled Kaggle environment and record actual commands, stdout/stderr, exit codes and fixes. Never invent successful execution.

**Pass:** reproducible manifests and an honest measured dataset/curriculum, not merely the nominal 10B allocations. **Current:** next 100M-input-cap CPU window `fe52fd65/1` is RUNNING after the real resume/cloud-backup gates passed. Inspect its final saved outputs before counting additional tokens or continuing; do not submit a duplicate while active.

### Phase E — checkpoint, loader and compute readiness

1. Verify Kaggle's runtime `HF_TOKEN` secret binding and private-repository upload/readback. Controller HF login does not satisfy this.
2. Validate padding/shifted-target accounting and wire the prepared assistant-only masks into the real SFT training consumer. The new preparation manifest explicitly keeps this consumer gate open.
3. Pin model/source/config/tokenizer/data-window identities in checkpoints.
4. Prove a full save → stop → resume path preserving weights, optimizer/scheduler/scaler where applicable, RNG, data cursor and token counts—not a weights-only restart.
5. Set wall-clock save/stop margins covering startup, downloads, evaluation and final checkpoint upload.
6. Notify Yash and obtain approval **before** any GPU pilot. Measure actual memory, supported precision, optimizer throughput, checkpoint overhead and available quota.
7. Estimate finish time from end-to-end measured token throughput and quota, not theoretical GPU throughput. If January 1 is infeasible, ask for a deadline/scope/compute decision before a long run; never promise the deadline without evidence.

**Pass:** loader, checkpoint durability/resume and measured feasibility gates, plus explicit training authorization.

### Phase F — foundation training

After approval, initialize fresh weights and train against the measured foundation curriculum, nominally 8.5B non-padding input tokens. Preserve periodic durable checkpoints and fixed validation. Count consumed real tokens, not EOS padding. Check divergence, data order, resume correctness and generation quality.

**Pass:** durable checkpoint and fixed-protocol foundation/code/FIM evidence. A lower training loss alone is insufficient.

### Phase G — code instruction, then tool/agent instruction

1. Use assistant-only loss for coding instructions/explanations, nominal allocation 1B input tokens.
2. Add tool/agent instruction, nominal allocation 0.5B input tokens, using the measured available data and explicitly documented repetition if later chosen.
3. Balance tool-call, no-call, clarification and recovery behavior. Keep a coding validation mixture so tool tuning does not erase code skills.
4. Compare each candidate against the previous checkpoint; do not promote a tool gain that hides a severe coding regression.

**Pass:** separate coding, explanation and tool execution outcomes. Prior small tool-SFT experiments were not promoted.

### Phase H — fixed evaluation and real serving/integration

- Freeze prompts, decoding, splits and evaluation rules before scoring candidates. Do not repeatedly tune on held-out benchmark tasks.
- Evaluate code/FIM correctness, explanation quality, tool selection, valid JSON/arguments, no-call precision, missed calls and end-to-end execution outcomes separately.
- Keep reference-solution/executor checks and isolated execution limits; never execute model-generated code on the controller PC.
- Replace `serving/nexus_bridge.py`'s placeholder with actual checkpoint/tokenizer loading and generation; verify cached inference against model forward behavior.
- Integrate real Nexus/Vision tool/action schemas and test the request → execute → result → next-action loop. Invalid calls must be surfaced, not replaced with fake results.
- Validate the cloud integration harness first. Phone/app deployment is a later explicit task, not permission to resume phone migration.

**Pass:** measured consumer-visible behavior with reproducible evaluation evidence and working inference, not hardcoded responses.

### Phase I — release/handoff

Publish only a validated candidate with pinned model/tokenizer identities, actual consumed/unique-data accounting, license/provenance limitations, evaluation protocol/results, inference usage and reproducible checkpoint/resume records. Keep legacy/reference checkpoints untouched. Do not describe an incomplete integration or an unverified run as a finished model.

## 9. File map

| File | Purpose |
|---|---|
| `CODEFORGE_PLAN.md` | This consolidated plan and new-PC handoff |
| `AGENTS.md` | Short agent-facing operating constraints |
| `README.md` | Project evidence/history and preparation usage |
| `configs/config_250M_fresh.yaml` | Architecture, training budgets and approval gate |
| `configs/fresh_data_production.yaml` | Pinned source mixture, sample/window limits and exclusion policy |
| `data/prepare_production.py` | Kaggle-only sampling/tokenization/packing/resume engine |
| `data/instruction_format.py` | Canonical instruction/tool text and assistant spans |
| `scripts/push_cpu_prep.py` | CPU-only source-snapshot submission and completed-output attachment |
| `scripts/kaggle_drive_backup.py` | Verified download of saved owner outputs; not a Drive uploader |
| `kaggle/check_hf_secret.py` | Notebook checkpoint-credential/readback gate |
| `training/`, `models/`, `evaluation/` | Training/runtime/evaluation implementation, still subject to readiness gates |
| `serving/nexus_bridge.py` | Currently unfinished serving integration |

## 10. Continue from another PC

### Option A — the exact same live session

Inside **this OMP chat**, type:

```text
/collab
```

OMP prints a private join command. On the other trusted PC, with OMP installed, run that exact command:

```text
omp join "<private join link printed by /collab>"
```

- This is the same live agent/session, not a new model independently guessing the context.
- The original PC and OMP process must stay running and connected. Agent tools and controller monitoring still run on the original PC; heavy workloads remain on Kaggle.
- No project/model-account authentication is needed on the guest merely to control the host agent; the host uses its own configured tools/accounts. OMP installation/network access on the guest are still required.
- The link grants control: treat it like a password, share it only with yourself, never put it in Git or this plan. Use `/collab view` only if read-only access is intended, and `/collab stop` when finished.
- Session traffic is end-to-end encrypted, but the guest receives sensitive conversation context. This session previously contained an authorization exposure; use only a trusted device and do not publish exports or links.
- Collaboration does **not** relocate files, logins or running processes onto the new PC. Do not start duplicate notebooks from a second controller.

### Option B — make the other PC the independent controller

1. Transfer the **current project folder**, including this plan, the new production modules/configuration and local changes. A GitHub clone alone is not proof those latest local files are present: the preparation notebook used an embedded source snapshot, and this preparation work has not itself pushed the new files to GitHub.
2. Do not copy the entire parent `own_llm` directory indiscriminately: `../.watch/accounts` and other private stores may contain credentials. Do not copy `.omp` account stores, `.kaggle` tokens, HF caches, rclone credentials or SSH keys as ordinary project files.
3. Yash installs/authenticates the needed controller CLIs on the new PC. Do not install a full local ML training stack merely to monitor Kaggle.
4. Open a terminal in the copied project folder and attach the plan:

```text
omp "@CODEFORGE_PLAN.md" "Read this handoff. Recheck the existing Kaggle preparation notebook; do not submit duplicate work or start model training."
```

This is **plan-based continuation**, not a claim that the entire previous chat history has migrated.

5. Recheck the exact notebook above before launching anything. The existing Kaggle run continues even if the original PC is turned off.
6. If migrating the current observer, transfer only `../.watch/observe-production.py` and `../.watch/fresh-production-launch-20261026-02.json` plus the small observation/verification reports you need. Its current folder layout expects a sibling `CodeForge-250M_win` directory; preserve/adapt that path before running it. These whitelisted files are not account authentication.
7. Establish new-PC monitoring before retiring the old controller. Do not run two independent submission/watchdog loops, revive the old account-rotation watchdog, or assume a transferred process ID is meaningful on the new machine.

### Full saved-chat migration: separate and security-sensitive

OMP stores resumable conversations under `~/.omp/agent/sessions/<encoded-project>/<timestamp>_<session-id>.jsonl`, with an adjacent artifact tree and sometimes referenced blobs under `~/.omp/agent/blobs/`.

A separately audited, safely transferred transcript can be opened from the new project directory using:

```text
omp --resume "<path to the reviewed session.jsonl>"
```

`--continue` selects local recent history; it does not download this conversation from GitHub. `/export` produces HTML for viewing, not a resumable chat backup. Authentication/configuration and local process state do not come with a transcript.

After resuming, check OMP's displayed working directory. A saved session can
restore its old directory when that directory exists; do not assume that copying
the transcript has rebased project paths, tools, observers or local artifacts.

**Do not blindly archive or publish the current raw session/artifacts:** historical tool output contained authorization material. This handoff deliberately does not export that raw transcript or copy credentials. Use the private live-session route or the plan-based route unless a security-reviewed saved-session transfer is separately prepared.

## 11. First instruction to the next agent

Read this file and `AGENTS.md` first. Recheck active CPU preparation `yashbajpai2027/codeforge-fresh-cpu-prep-fe52fd65/1`; do not submit a duplicate. Latest completed state is `d4d37022/1`, parent `2d372b23/1`: 20 audited windows, 10,065,678 train IDs plus 101,480 validation IDs, same frozen tokenizer `3b06dbef...`. Separate private Kaggle backup `codeforge-fresh-prepared-w20-20261026/1` passed complete 184-file cloud readback. Use the exact completed state and retain the full parent chain; never restart tokenizer fitting or count-add legacy `534125d9` windows. The native Parquet prefetch shutdown hang is fixed and both real CLI jobs passed 76 tests and normal exit. Drive cloud sync, Kaggle `HF_TOKEN`, SFT mask consumption and training feasibility remain open. Do not change accounts, perform logins or start GPU/model training; notify Yash and obtain explicit approval before any training.
