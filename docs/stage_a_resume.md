# Audited Stage A resume

This trainer loads the exact number of rows in each tokenized shard. It does not
pad short shards, regenerate data, resize the tokenizer or wrap the dataset.
The audited resume path currently supports one GPU only.

## New-format checkpoints

Use `python training/train.py --resume --max_hours 6` after staging the intended
checkpoint, tokenizer and shards. Schema-2 resume fails if the stored
data/tokenizer/config fingerprint, split membership, valid-length/loss-mask
hashes, batch geometry or optimizer counters do not match.

Checkpoints include the actual sample cursor, cumulative real input IDs
(`resume_state.input_tokens_seen`), optimizer-applied update count, GradScaler
state and RNG state. The real-ID count must match the saved ordered cursor.
The schedule step and optimizer-applied count are separate because
mixed-precision overflow can skip an optimizer update. A final short batch is
normalized by its real sample count and saved. `data.target_tokens` or
`--max-input-tokens` caps cumulative real IDs, not padded slots; whole rows that
would exceed the cap are left unconsumed. See the README's foundation consumer
section for production-window selection and input-vs-target semantics.

## Legacy checkpoint migration

A legacy checkpoint without resume metadata, or with schema-1 metadata,
requires an explicit audited JSON record passed with `--legacy-migration PATH`.
Inspect `training/train.py` for the required fields. Bind the record to the exact
checkpoint hash, schema-2 data/tokenizer/config fingerprint, shard order,
reconstructed cursor and real-input count. Never infer the cursor or cumulative
tokens from the reported checkpoint step alone. Schema-1 migration cannot
change the saved schedule step; legacy unfiltered cursors retain their existing
schedule/permutation reconciliation checks.

`training/resume_safety.py` filters nonexistent positions from the old seeded
permutation while keeping valid examples in their original relative order.
This avoids feeding uninitialized rows going forward. It cannot undo or prove
harmless any bad batches consumed before the migration.

## Stop and save

`--max_hours` limits training-loop wall time. Optionally set
`CF_ABSOLUTE_STOP_EPOCH` to a Unix timestamp. At each batch boundary the trainer
checks both limits and starts no new batch after either is reached. A batch or
validation already underway may finish first. Allow additional time for final
checkpoint serialization and any external upload; this is not a hard process
kill or an upload deadline. `STOP_AND_SAVE` in the repo directory also requests
a clean stop after the current batch.

`--session-steps N` runs a bounded smoke test without changing the LR schedule.
Use `CF_TELEMETRY_DISABLED=1` to disable the existing telemetry sender in private
runs. Keep credentials in the environment or the platform's secret store, never
in notebook source or committed files. Uploads and platform GPU-session cleanup
are managed by the runner, not this trainer.

## Tests

Run `python -m unittest discover -s tests -v` with the project dependencies
installed. CPU tests cover shard checks, fingerprint changes, legacy cursor
reconciliation, filtered ordering, checkpoint state, and partial-batch scaling.
