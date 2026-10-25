# CodeForge preflight runbook (CPU, read-only)

Purpose: confirm file identity and internal consistency before any resume. It does not measure model quality.

## Inputs (all mandatory, none optional)
ckpt_sha256, ckpt_bytes, train_py_sha256, tokenizer_sha256, n_train_shards, n_valid_sidecars, seqs_per_step, plus explicit globs for ckpt, train.py, tokenizer, train shards, val shards.

## Order of checks
1. Mounts found (each glob resolves).
2. train.py sha256 matches the audited source.
3. Data layout: every train and val shard loaded, hashed, id-range checked; every sidecar checked; tokenizer sha; fingerprint computed.
4. Checkpoint size, then sha256. A mismatch stops here and nothing is loaded.
5. Only then torch.load(weights_only=True). If the safe loader fails, report the exact error. There is no unsafe fallback.
6. Counters read: step, sample_cursor, completed_updates, optimizer_step_count, resume_state agreement; tied weights counted once; all state_dict tensors finite.

## What a pass means
Identity and consistency against the audited lineage (train.py sha 5e0ec263...). Counter and cursor relations are tied to that train.py, not universal rules. The memory estimate is uncalibrated and does not gate.

## Expected identity rejection
A different file (for example the periodic step-420 copy, a different size and sha) fails the size/sha precondition and is not loaded. That is the check working as designed, not a new finding about the file.

## Status labels
- Fixture-tested: local synthetic tests pass.
- Real-validated: run on the mounted private dataset and checkpoint with CPU only, internet off, no secrets, session stopped afterwards.

## Real run recorded
latest_checkpoint.pt passed (step 420, cursor 107520, counters 420/420). Data fingerprint matches the previously recorded value. Periodic copy rejected by identity.
