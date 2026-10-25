# Continuation to step 462: status note (25 Oct 2026)

This note records what was run and measured after the Stage A gate (step 100). It makes no claim that the model is a reliable coding model.

## Final checkpoint
- File: `proj/checkpoints/gate/latest_checkpoint.pt` (Kaggle private dataset `cf-finish462-output`)
- SHA-256 `f0d518fd96266178b3772bbd5b461df718aec83172f0fceb64a1d921c448ed53`, 2,951,299,239 bytes
- step 462, sample_cursor 118125 of 118125, completed_updates 462, 147 state tensors
- Verified by hash and counters at the start of the step-462 eval run.

## Training
- 245,924,864 parameters, 16 layers, hidden 1024, 16 heads, 4 KV heads, FFN 2816, vocab 32,000, context 2048, untied embeddings.
- Continuation: lr 1e-4 cosine to 1e-5, warmup 50, 462 updates, batch 8 x 32 = 256 sequences. One pass over the data; the last update was partial (109 sequences).
- Resume chain: gate step 100, then step 420 (audited resume), then step 462.
- Logged tokens count input slots (rows x 2047), not valid-length-masked tokens.

## Validation numbers (keep the provenance)
- Stored `val_loss` in the step-462 file, 1.6887902, is the in-loop value from step 450 carried forward. It is not a fresh step-462 eval.
- Fresh harness held-out loss (216 of 221 rows, fp16): step 420 = 1.690060, step 462 = 1.688380.

## Matched test results (reporting only, never used for selection)
Same harness (`evaluation/cf_score.py`, SHA-256 6429ee43...), prompts, greedy decoding, 320-token cap. Generated programs were exported on Kaggle and never executed there.
| Benchmark | A2 | gate (100) | step 420 | step 462 |
| --- | --- | --- | --- | --- |
| HumanEval (164), passed | 7 | 6 | 5 | export done (164/164), isolated scoring pending |
| MBPP 2-shot (500), passed | 5 | 5 | 9 | export done (500/500), isolated scoring pending |
| Fresh tasks (24), passed | 0 | 1 | 0 | export done (24/24), isolated scoring pending |

Counts are small and descriptive. Earlier paired comparisons were not significant.

## Limits
- HumanEval and MBPP are public; overlap with pretraining text was not audited.
- The 24 fresh tasks are a small custom set.
- The 162-of-164 cap-hit finding applies to gate raw outputs, not step 420.
- No separate validation set for model selection exists.

## What is in this change
- `training/train.py`: the audited version (valid-length loss masking, `CF_VAL_DIR` held-out path, DDP guard). SHA-256 5e0ec263...
- `tools/preflight/`: CPU preflight (identity, shard and sidecar checks, data fingerprint), fixture tests, runbook. Fixture tests only prove the logic; the preflight was also run on the real step-420 data and checkpoint.
- `evaluation/`: interface audit with tests, and the unchanged scoring harness `cf_score.py`.
- `kaggle/continuation/`: the training and eval notebook cells used for steps 420 and 462.
- `training/utils.py` is intentionally unchanged: the audited copy differs from this repo only in line endings.
