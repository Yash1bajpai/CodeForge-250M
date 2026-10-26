# CodeForge project context

## Purpose and current run

CodeForge is a small code-model research project intended for Nexus/Vision integration. `serving/nexus_bridge.py` remains a placeholder: do not describe working inference or tool execution as implemented.

The active fresh plan is `fresh-20261025`: random initialization, 245,924,864 parameters (~246M), target 10B non-padding input tokens by January 1, 2027. The 8.5B foundation / 1B code-instruction / 0.5B tool-instruction split is a planning budget, not available-data or quality evidence. Preserve the completed legacy Stage A step 2,898 and continuation checkpoints; do not resume them into this fresh model.

## Layout

| Path | Responsibility |
|---|---|
| `configs/config_250M_fresh.yaml` | Fresh model/training plan and explicit approval gate |
| `configs/fresh_data_pilot.yaml`, `data/prepare_fresh.py` | Bounded CPU preparation pilot |
| `scripts/phone_shell.sh`, `scripts/setup_phone.sh` | Native Termux Debian launcher and Debian controller installation |
| `scripts/phone_control.py` | Phone status and explicit Drive probe; no training launch |
| `scripts/kaggle_drive_backup.py` | Owner-only saved-output download to a fresh local staging directory |
| `scripts/push_cpu_prep.py` | Explicit CPU preparation submission, not authorized by migration |
| `kaggle/` | Cloud notebook sources; default metadata can target training |
| `training/`, `models/`, `evaluation/` | Cloud ML execution and evaluation |
| `serving/` | Unfinished serving integration |
| `tests/` | Existing unittest-style checks |
| `docs/phone-migration.md` | Phone checklist, evidence, operations and outstanding gates |
| `docs/stage_a_resume.md`, `docs/CONTINUATION_STEP462.md` | Historical continuation notes, not the fresh-run plan |

## Settled operation and safety

- The phone is a controller, not a training/data-processing host. Native Termux launches Debian with `codeforge` (`scripts/phone_shell.sh`); source, cloud credentials and evidence directories are shared into Debian. Workspace: `~/projects/own_llm/CodeForge-250M_win`; external control state: `~/projects/own_llm/.watch`; staging/evidence: `~/codeforge-storage`.
- Only Kaggle owner `yashbajpai2027` is enabled. Large datasets/checkpoints stay cloud-side. HF checkpoint repository: `Yash1bajpai/CodeForge-250M-rotation`.
- Do not launch GPU jobs, model training, or additional CPU preparation during migration. Notify the owner before any GPU/training launch and obtain explicit approval. `CF_TRAINING_APPROVED=1` belongs only on an explicitly approved cloud training invocation, never in phone startup/environment files.
- Keep secrets outside Git and never print values. Kaggle owner token: `~/.kaggle/access_token`; rclone config: `~/.config/rclone/rclone.conf`; SSH keys/authorized keys: `~/.ssh/`; optional private environment file: `~/.config/codeforge/control.env`.
- Backups download to fresh LOCAL staging first. `kaggle_drive_backup.py` does not upload to Drive. Use `rclone copy`, never `sync`, followed by explicit cloud readback/check evidence.
- Preserve intentional uncommitted work and legacy external `.watch` scripts. The historical Windows rotation/watchdog scripts must not auto-start on the phone.
- Do not install the full ML `requirements.txt` in Termux. Use the phone-specific dependency instructions.

## Evidence and remaining gates

Verified CPU pilot: `yashbajpai2027/codeforge-fresh-cpu-preparation-139cbd2a`, version 1 COMPLETE; 3,272 train / 61 validation documents, 6,787,976 / 182,325 stored real token IDs. Candidate tokenizer: 32k entries, unpromoted. On October 26, 2026 the phone recovered and SHA-256-verified all 24 producer files / 45,759,067 bytes, then compared downloaded Drive copies: 24 matches, zero differences. Evidence: `~/codeforge-storage/logs/pilot-cloud-verification.json`.

`yashbajpai2027/codeforge-hf-readiness-8cd44a26` completed, but Kaggle Secrets failed with `ConnectionError` / `backend-request-failed`. A working PC HF key does not prove Kaggle checkpoint access. Official Kaggle CLI docs provide no secret-binding support; use the notebook Secrets UI. No model training has been launched for the fresh run.

Production gates remain: representative licensed corpus/mixture, bounded-memory full-scale preparation, contamination coverage, promoted production tokenizer, valid-token/instruction-loss accounting, checkpoint upload/readback from Kaggle, measured training throughput/compute feasibility, and fixed evaluation protocol. Phone installation, key-only USB SSH, GitHub/Kaggle/HF access, an authenticated coding-agent call, HF write/readback, Drive OAuth and cloud readback are verified. See `docs/phone-migration.md` and private `~/codeforge-storage/logs/phone-migration-receipt.json`; do not republish credential or account-identity files.

## Deeper notes and checks

Start with `README.md` and `docs/phone-migration.md`; do not duplicate long evidence logs here. Non-secret evidence belongs under `~/codeforge-storage/logs/`, with summaries in the migration handoff. Logs must exclude signed URLs and credentials.

Existing checks include `python -m unittest discover -s tests`. Exercise changed controller paths without installing the ML stack on the phone. No setup check may launch training or preparation.
