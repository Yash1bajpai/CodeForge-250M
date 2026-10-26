# Phone-first control plane migration

## Architecture and limits

The Android Termux phone is the verified controller as of October 26, 2026. Windows is not required for normal phone-side operation. Keep the Windows copy as a preserved backup; this migration deletes no legacy checkpoints. Large datasets, training and checkpoints remain in Kaggle/Hugging Face/Drive; the phone holds source, status reports and bounded temporary staging, not the ML stack.

The native Termux shell provides key-only USB SSH and `proot-distro`. The coding agent and cloud clients run inside Debian: OMP publishes Linux ARM64 native bindings, not Android bindings. `scripts/phone_shell.sh`, installed as `codeforge`, binds the project, evidence and private provider directories into Debian. `~` in the commands below refers to the active environment's home; these bound directories refer to the same files in either environment. Debian's package/runtime files stay inside Termux app storage.

- Repository: `~/projects/own_llm/CodeForge-250M_win`.
- External controller state/reference scripts: `~/projects/own_llm/.watch`.
- Local staging and non-secret evidence: `~/codeforge-storage/{datasets,checkpoints,logs}`.
- Kaggle: only `yashbajpai2027`; HF: `Yash1bajpai/CodeForge-250M-rotation`.
- Intended Drive remote/base: `codeforge-drive:codeforge-storage`.
- Historical `.watch/rotation.py` and `watchdog.py` are preserved references, not phone daemons. They contain Windows/old Stage A assumptions; never auto-start them.

No additional CPU preparation, GPU pilot or model training is authorized by this migration. `push_cpu_prep.py` can submit a fresh CPU preparation only after a separate request; do not run it now. Notify the owner before any GPU/training launch and obtain explicit approval. Never set `CF_TRAINING_APPROVED=1` in shell startup files, tmux startup, service configuration or phone setup. Only an explicitly approved cloud training invocation may receive that flag.

## Current evidence, not future promises

The active fresh plan is `fresh-20261025`, ~246M random-weight parameters, 10B target tokens and January 1, 2027 deadline. The 8.5B / 1B / 0.5B foundation/code-instruction/tool-instruction allocations are planning budgets only. Legacy Stage A step 2,898 and continuation checkpoints are preserved, not resumed as fresh-model initialization.

Verified CPU pilot: [`yashbajpai2027/codeforge-fresh-cpu-preparation-139cbd2a`](https://www.kaggle.com/code/yashbajpai2027/codeforge-fresh-cpu-preparation-139cbd2a), version 1 COMPLETE. It produced 3,272 training documents / 61 validation documents and 6,787,976 / 182,325 real stored token IDs. The 32k tokenizer remains a candidate, not a production promotion. The phone independently recovered and SHA-256-verified all 24 producer files / 45,759,067 bytes, then downloaded the Drive copies for comparison: **24 matching files, zero differences**. Evidence is in `~/codeforge-storage/logs/pilot-cloud-verification.json` and `pilot-cloud-check-combined.txt`. The original Windows backup remains at `G:/My Drive/codeforge-storage/datasets/fresh-20261025-pilot01`.

The PC HF token works, but the completed readiness notebook [`yashbajpai2027/codeforge-hf-readiness-8cd44a26`](https://www.kaggle.com/code/yashbajpai2027/codeforge-hf-readiness-8cd44a26) could not obtain the Kaggle secret (`ConnectionError`, `backend-request-failed`). It did not prove Kaggle checkpoint writes. [Official Kaggle CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#using-secrets-in-kernels) explicitly has no CLI secret-binding support; notebook Secrets UI authorization is required. Do not embed the key in notebook source as a workaround. No fresh model training has launched.

## Migration checklist and evidence ledger

The following checks were exercised on the real phone; production-model readiness remains separate.

- [x] Preserve current source, including intentional unpublished fresh-pilot changes. The initial 85-file source snapshot passed phone-side SHA-256 comparison; Git history is also preserved as a bundle.
- [x] Preserve external `.watch` history/reference files and `STOP_WATCHDOG`; do not auto-start legacy Windows control scripts.
- [x] Install/open stable F-Droid Termux on Android 16, ARM64 device I2219. Approximately 46 GiB remained after the runtime installation.
- [x] Install native Termux OpenSSH, tmux and proot-distro; install Debian 13 ARM64 and execute the complete `scripts/setup_phone.sh` successfully.
- [x] Transfer the project to the shared portable workspace without a Windows environment or the ML dependency stack.
- [x] Verify key-only SSH as `u0_a299`, listening only on `127.0.0.1:8022`; password and keyboard-interactive login are disabled.
- [x] Verify USB access with `adb forward tcp:28022 tcp:8022` and the PC's `ssh codeforge-phone-usb` alias. SSH is for remote assistance, not a requirement for direct phone use.
- [x] Transfer credentials outside Git with mode 600 files/mode 700 directories. Verify GitHub, owner Kaggle, HF identity and private HF repository read/write/readback; remove the own HF probe.
- [x] Authorize the correct 5 TB Drive using a dedicated OAuth client and pass upload/download SHA-256 comparison plus own-probe deletion.
- [x] Read the existing completed pilot/readiness notebook status and live quota without submitting a notebook.
- [x] Verify the full pilot bundle against Drive downloads: 24 matching producer files, zero differences.
- [x] Exercise native tmux session `codeforge`: attach, run a command with training approval false, detach, and confirm the session survives. Apply the Termux-specific battery exemption/background permission and request wake-lock.
- [x] Record the cutover receipt and remaining production gates. The phone controller works without Windows; a smoke check is not a guarantee of unattended Android uptime.

### Verification updates

Evidence under `~/codeforge-storage/logs/`: `phone-migration-receipt.json`, `phone-status-resume01.json`, `drive-probe-custom01.json`, `pilot-cloud-verification.json`, `pilot-cloud-check-combined.txt`, and `phone-hf-readback.json`. The receipt retains account identity details privately; no credential values belong in these reports.

Runtime: Python 3.13.5, Bun 1.4.3, OMP 18.12.6 with Linux ARM64 bindings, rclone 1.75.2. The pinned installer was exercised end to end. OMP made a real authenticated no-tool model call and correctly stated the loaded training rule and launcher name.

Drive uses the dedicated **CodeForge Storage Controller** Google Cloud project `numeric-polygon-512804-d7`, with only the required Drive scope declared for this client. Its OAuth app is **In production**, avoiding Testing's seven-day grant expiry, but is **not Google-verified** and remains subject to Google's unverified-app policies/user cap. No billing account, VM or paid compute resource was created. The [public privacy disclosure](storage-privacy.md) is published in the existing project repository. Do not revert to rclone's shared client: it produced HTTP 403 project-wide API quota errors and is being retired.

The original project conversation was privately restored through `omp session pack/unpack` on the phone. Private conversation archives and credentials stay outside Git.

## Credentials and minimal environment

| Purpose | Private location / contract |
|---|---|
| Kaggle owner token | `~/.kaggle/access_token` |
| HF token | `HF_TOKEN` supplied privately, or standard cache `~/.cache/huggingface/token` (`HF_HOME` may override) |
| Optional private environment file | `~/.config/codeforge/control.env`; explicit loading only, no training approval flag |
| Drive OAuth configuration | `~/.config/rclone/rclone.conf` |
| SSH | `~/.ssh/authorized_keys` and private keys outside the repository |

Use mode 700 for private configuration directories and 600 for secret/config/key files as appropriate. Never commit an environment file or paste token values into commands retained in shell history. Transfer secrets over the established private channel; source archives must exclude them.

Use `scripts/setup_phone.sh` inside Debian to install controller tools and `scripts/requirements-phone.txt`, not root `requirements.txt`. Normal dependency resolution is supported inside Debian, including Linux wheels; native Termux is not the controller's Python environment. The phone does not need torch, deepspeed, transformers, datasets, training tokenizer builds or GPU packages. Native Termux's `hf-xet`/other compiled dependency issues are avoided rather than silently replacing those libraries.

## Phone operations (after installation)

Run `codeforge` from native Termux to enter the project in Debian with the controller virtual environment on `PATH`. Run the following from `~/projects/own_llm/CodeForge-250M_win`. Report paths must be fresh, and parent directories must exist. `scripts/phone_shell.sh` refuses an inherited training-approval flag; `scripts/setup_phone.sh` likewise refuses one and never submits a job.

For direct use, open Termux and run `tmux attach -t codeforge` to enter the prepared Debian project shell. If Android has killed that session, use `tmux new -s codeforge codeforge` instead. `codeforge` alone also enters Debian. Google Antigravity is authenticated and has passed a live `google-antigravity/gemini-3.8-flash` request; start it with `omp --model google-antigravity/gemini-3.8-flash --continue`. Check private `~/codeforge-storage/logs/provider-verification.json` before using an authorization marked for revocation or re-login. No command here launches training.

If USB assistance is needed after Termux restarts, run `sshd -f ~/.ssh/codeforge_sshd_config` in native Termux, then recreate the PC forwarding rule above. Do not expose SSH to Wi-Fi just to restore a USB connection.

```sh
python scripts/phone_control.py status --report "$HOME/codeforge-storage/logs/phone-status-$(date -u +%Y%m%dT%H%M%SZ).json"
python scripts/phone_control.py drive-probe --remote codeforge-drive:codeforge-storage --report "$HOME/codeforge-storage/logs/drive-probe-$(date -u +%Y%m%dT%H%M%SZ).json"
```

Successful HF metadata access from the phone proves only phone-side access, not Kaggle Secrets availability or checkpoint upload/readback from a training notebook. A required integration failure returns nonzero; retain the JSON evidence and do not reinterpret it as training readiness.

`status` inspects existing jobs/authentication; it does not submit training or preparation. A listed rclone remote does not prove OAuth/cloud access. `drive-probe` is an explicit storage write/readback operation, not a training launch; retain its SHA-256 cloud readback result. Use a new report filename for later runs.

For a fresh local download of the existing pilot's saved outputs (not a new preparation):

```sh
python scripts/kaggle_drive_backup.py --kernel yashbajpai2027/codeforge-fresh-cpu-preparation-139cbd2a --version 1 --destination "$HOME/codeforge-storage/datasets/fresh-20261025-pilot01-phone"
```

The destination must not already exist. The helper verifies owner/current saved version, downloads bounded chunks and writes `backup-manifest.json` only after local SHA-256 readback. It **does not upload to Drive**, schedule recurring backup or recover unsaved files from killed jobs. Ensure enough free phone storage before downloading; large production artifacts should not be mirrored wholesale to the phone.

After local backup verification and OAuth, explicitly copy to Drive without deleting destination files:

```sh
rclone copy "$HOME/codeforge-storage/datasets/fresh-20261025-pilot01-phone" codeforge-drive:codeforge-storage/datasets/fresh-20261025-pilot01 --immutable --checksum --exclude /backup-manifest.json
rclone check "$HOME/codeforge-storage/datasets/fresh-20261025-pilot01-phone" codeforge-drive:codeforge-storage/datasets/fresh-20261025-pilot01 --download --one-way --exclude /backup-manifest.json
```

Never use `rclone sync`. `--immutable` refuses changed existing files instead of overwriting them. A conflicting destination needs investigation, not a forced overwrite. The `check --download` step reads cloud bytes for comparison; preserve its successful result and compare the expected 24 producer files/45,759,067 bytes separately from the downloader's additional backup manifest. A small `drive-probe` passing does not certify this dataset bundle. Keep local staging until bundle-specific cloud proof is recorded; do not mistake Drive-for-desktop local readback for remote readback.

`backup-manifest.json` is a per-download local audit record with its own timestamp; it is excluded so a second verified download does not conflict with the original backup's audit record. Keep each local manifest and store cloud verification evidence separately under `logs/`.

Use `--checksum` with immutable copies: independently downloaded copies have different local modification times even when their contents are identical. Content mismatches must still fail; do not force an overwrite to fix a timestamp mismatch.

## Repeatable provider and service checks

Run these from the Debian project shell, not inside the coding agent's message editor:

```sh
gh api user --jq .login
omp --no-session --no-tools --model google-antigravity/gemini-3.8-flash --max-time 90s -p 'Reply only AGY_OK.'
python scripts/phone_control.py status
python scripts/phone_control.py drive-probe --remote codeforge-drive:codeforge-storage --report "$HOME/codeforge-storage/logs/drive-check-$(date -u +%Y%m%dT%H%M%SZ).json"
```

Expected: GitHub identity `Yash1bajpai`, an actual `AGY_OK` model response, required status checks successful, and Drive `cloud_readback_verified: true` plus `own_object_removed: true`. A model catalog listing or cached credential's existence is not proof that the provider accepts it.

After any credential exposure, have the account owner revoke the affected provider authorization through its security controls before signing in again. Local credential deletion or a new login alone does not prove the old authorization was revoked. For a fresh Codex login, use `omp login openai-codex`; only after revocation/re-login, test `omp --no-session --no-tools --model openai-codex/gpt-6.1-sol --max-time 90s -p 'Reply only CODEX_OK.'`. Never run or paste raw `omp auth export` output as a status check: it contains live credentials.

These checks establish current access, not permanent availability. Provider grants can expire or be revoked, quotas can run out, networks can fail, and Android can terminate Termux. Keep private evidence under `~/codeforge-storage/logs/`; model-training readiness still requires the separate production gates below.

## Long-running controller sessions

Use `tmux new -s codeforge` for interactive work and `tmux attach -t codeforge` after reconnecting. Request `termux-wake-lock` while required, then `termux-wake-unlock` when finished. Configure Android battery optimization/background permissions deliberately and record what was actually set. tmux survives SSH disconnection but not Termux/Android process termination, reboot, revoked permissions, network loss or power loss; wake-lock is not a service guarantee. Retain logs and use idempotent on-demand status/backup operations, not legacy Windows watchdogs. No boot hook should start training or extra preparation.

## Production gates still open after migration

Phone readiness does not promote the CPU pilot. Required gates include representative production data and licensing/source access; bounded-memory deduplication/loading and full-scale preparation measurements; contamination coverage; production tokenizer freeze; valid-token accounting and instruction loss masking; Kaggle checkpoint credential upload/readback; measured training throughput/compute feasibility against the deadline; and fixed evaluation criteria. Nexus/Vision tool schemas, selection/argument/execution evaluation and real serving still need implementation: `serving/nexus_bridge.py` is a placeholder, not a working tool-calling bridge. See the README for model history and pilot limits.
