# CodeForge project context

## Purpose and current run

CodeForge is a small code-model research project intended for Nexus/Vision integration. `serving/nexus_bridge.py` remains a placeholder: do not describe working inference or tool execution as implemented.

The active fresh plan is `fresh-20261025`: random initialization, 245,924,864 parameters (~246M), target 10B non-padding input tokens by January 1, 2027. The 8.5B foundation / 1B code-instruction / 0.5B tool-instruction split is a planning budget, not available-data or quality evidence. Preserve the completed legacy Stage A step 2,898 and continuation checkpoints; do not resume them into this fresh model.

## Layout

| Path | Responsibility |
|---|---|
| `configs/config_250M_fresh.yaml` | Fresh model/training plan and explicit approval gate |
| `configs/fresh_data_pilot.yaml`, `data/prepare_fresh.py` | Bounded CPU preparation pilot |
| `configs/fresh_data_production.yaml`, `data/prepare_production.py` | Pinned Kaggle-only production recipe, frozen tokenizer and resumable windows |
| `data/instruction_format.py` | Canonical instruction/tool traces with assistant-only character spans |
| `scripts/phone_shell.sh`, `scripts/setup_phone.sh` | Native Termux Debian launcher and Debian controller installation |
| `scripts/phone_omp.sh`, `scripts/phone_check.sh`, `scripts/install_phone_commands.sh` | Native Termux `omp`/`cf-check` commands and installation |
| `scripts/phone_control.py` | Aggregate `check`, status and explicit Drive probe; no training launch |
| `scripts/kaggle_drive_backup.py` | Owner-only saved-output download to a fresh local staging directory |
| `scripts/push_cpu_prep.py` | Explicit CPU preparation submission, not authorized by migration |
| `kaggle/` | Cloud notebook sources; default metadata can target training |
| `training/`, `models/`, `evaluation/` | Cloud ML execution and evaluation |
| `serving/` | Unfinished serving integration |
| `tests/` | Existing unittest-style checks |
| `docs/phone-migration.md` | Phone checklist, evidence, operations and outstanding gates |
| `docs/stage_a_resume.md`, `docs/CONTINUATION_STEP462.md` | Historical continuation notes, not the fresh-run plan |

## Settled operation and safety

- The Windows PC is the primary controller: edit/sync source and configs, submit/monitor notebooks and coordinate backups. ALL corpus processing, tokenizer fitting, tokenization/packing, model training and evaluation run on Kaggle, never the PC or phone. The phone installation is preserved but not the active workflow.
- Only Kaggle owner `yashbajpai2027` is enabled. Large datasets/checkpoints stay cloud-side. HF checkpoint repository: `Yash1bajpai/CodeForge-250M-rotation`.
- The owner authorized Kaggle CPU data preparation on October 26, 2026. This is NOT model/GPU-training approval. Notify the owner before any GPU/training launch and obtain explicit approval. `CF_TRAINING_APPROVED=1` belongs only on an approved cloud training invocation, never startup/environment files. Authentication is owner-controlled; do not resume login setup unprompted.
- Keep secrets outside Git and never print values. Kaggle owner token: `~/.kaggle/access_token`; rclone config: `~/.config/rclone/rclone.conf`; SSH keys/authorized keys: `~/.ssh/`; optional private environment file: `~/.config/codeforge/control.env`.
- Backups download to fresh LOCAL staging first. `kaggle_drive_backup.py` does not upload to Drive. Use `rclone copy`, never `sync`, followed by explicit cloud readback/check evidence.
- Preserve intentional uncommitted work and legacy external `.watch` scripts. The historical Windows rotation/watchdog scripts must not auto-start on the phone.
- Do not install the full ML `requirements.txt` in Termux. Use the phone-specific dependency instructions.
- Honor the Codex revocation/fresh-login gate in private `~/codeforge-storage/logs/provider-verification.json`. An imported credential or successful local login does not prove a compromised authorization was revoked.

## Evidence and remaining gates

Verified CPU pilot: `yashbajpai2027/codeforge-fresh-cpu-preparation-139cbd2a`, version 1 COMPLETE; 3,272 train / 61 validation documents, 6,787,976 / 182,325 stored real token IDs. Candidate tokenizer: 32k entries, unpromoted. On October 26, 2026 the phone recovered and SHA-256-verified all 24 producer files / 45,759,067 bytes, then compared downloaded Drive copies: 24 matches, zero differences. Evidence: `~/codeforge-storage/logs/pilot-cloud-verification.json`.

`yashbajpai2027/codeforge-hf-readiness-8cd44a26` completed, but Kaggle Secrets failed with `ConnectionError` / `backend-request-failed`. A working PC HF key does not prove Kaggle checkpoint access. Official Kaggle CLI docs provide no secret-binding support; use the notebook Secrets UI. No model training has been launched for the fresh run.

Production recipe: 5.95B multilingual GitHub-code + 1.275B clean Python + 1.275B FineWeb-Edu foundation targets; OpenCoder diverse/real-user instruction targets 0.8B/0.2B; Glaive/Hermes tool targets 0.3B/0.2B. All revisions/license allowlists are pinned in YAML. These are training allocations, NOT proven unique supply; especially the small tool corpora cannot establish 0.5B unique tokens. Never repeat rows silently. Prepare each unique admitted record once and report source exhaustion.

`push_cpu_prep.py --config configs/fresh_data_production.yaml` submits private CPU-only windows (four-hour/roughly100M-input-token bounds). `--verify-only` runs contracts/live-source preflight; `--tests-only` isolates contracts, all on Kaggle. `--resume-kernel` attaches an exact COMPLETE owner notebook version and pins its runtime image; ERROR inputs are rejected. Explicit `--tokenizer-dataset owner/slug/version` recovery validates/adopts only frozen tokenizer/sample into fresh v2 state, never old SQLite/windows/totals; it cannot mix with resume/diagnostic modes. The backup helper can preserve COMPLETE/ERROR saved outputs, never active notebooks. V2 binds recipe/code/runtime/tokenizer identities to atomic SQLite state/cursors/dedup metadata. Parquet prebuffering is disabled and resources close before completion. SFT loss-mask consumption remains a separate training gate.

Latest completed real chain: `yashbajpai2027/codeforge-recovered-cpu-prep-2d372b23/1` → `codeforge-fresh-cpu-prep-d4d37022/1`. Both passed 76 tests and normal real CLI exit. Independent audit verified 20 windows, 8,330 distinct normalized documents, 10,065,678 train IDs + 101,480 validation IDs, four byte/control round trips and unchanged frozen tokenizer `3b06dbef...`.

Private dataset `yashbajpai2027/codeforge-fresh-prepared-w20-20261026/1` holds both complete backups; separate cloud readback checked all 184 producer files / 195,085,969 bytes by length/SHA-256. Drive copies are mounted-readback verified, NOT independently cloud-sync verified; managed Drive browser requires sign-in and no login was attempted.

Active next CPU preparation: `yashbajpai2027/codeforge-fresh-cpu-prep-fe52fd65/1`, latest RUNNING, 100M additional-input-token cap. Inspect `../.watch/scaled-preparation-observed.json` and the actual saved output before counting new data or resubmitting. Local service `CodeForgeCPUScaleObserver` uses `../.watch/observe-cpu-prep.py` and does not auto-resubmit.

Original `534125d9/1` ended ERROR after preparation due native Parquet/HF range I/O surviving main return; its 93 files were preserved and audited. Old state/shards remain unpromoted and must not be count-added or silently resumed; tokenizer-only adoption avoids refitting. Kaggle `HF_TOKEN` remains unavailable. No GPU/model training started; owner approval, SFT-mask consumer and feasibility gates remain open. Full evidence and continuation instructions are in `CODEFORGE_PLAN.md`.

Production gates remain: representative licensed corpus/mixture, bounded-memory full-scale preparation, contamination coverage, promoted production tokenizer, valid-token/instruction-loss accounting, checkpoint upload/readback from Kaggle, measured training throughput/compute feasibility, and fixed evaluation protocol. Phone installation, key-only USB SSH, GitHub/Kaggle/HF access, an authenticated coding-agent call, HF write/readback, Drive OAuth and cloud readback are verified. See `docs/phone-migration.md` and private `~/codeforge-storage/logs/phone-migration-receipt.json`; do not republish credential or account-identity files.

## Deeper notes and checks

Start with `CODEFORGE_PLAN.md` for the consolidated plan, current unverified production run and another-PC handoff, then `README.md` for evidence/history. `docs/phone-migration.md` is historical phone setup, not the active controller workflow. Do not duplicate long logs here. Current PC evidence is under `../.watch/`; historical phone evidence remains in `~/codeforge-storage/logs/`. Logs must exclude signed URLs and credentials.

Run the existing suite on Kaggle through `python scripts/push_cpu_prep.py --config configs/fresh_data_production.yaml --tests-only --report <fresh-report-path>`. Controller-only guards may be exercised locally, but no corpus/tokenizer/model workloads belong on the PC or phone. Setup checks must not launch preparation or training.
