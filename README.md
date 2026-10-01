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
- Stage B and tool-use SFT have not been completed. The 10B-token goal is a future milestone, not the amount already trained.

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

The pipeline downloads sources, filters text, deduplicates, tokenizes and packs 2,048-token uint16 sequences. Its current limitations matter:

- [`data/deduplicate.py`](data/deduplicate.py) uses normalized MD5 hashes and two ordered-shingle bucket hashes. It is **not true MinHash/LSH near-deduplication**, and does not establish a unique-token corpus.
- [`data/filter_quality.py`](data/filter_quality.py) applies Python AST checks to selected pure-code sources. Its character-based bracket check is not a universal syntax validator and can reject valid Python containing brackets inside strings or comments.
- [`data/tokenize_dataset.py`](data/tokenize_dataset.py) attempts 50% FIM transformation on eligible samples from **every source**, including prose, instruction and tool data. This is a known format problem, not a validated tool-training curriculum. The next data build should restrict FIM to code and leave chat/tool examples intact.
- No completed benchmark-decontamination audit is reported here. Planned contamination checks must not be mistaken for checks already run.

The nominal batch is 8 sequences × 32 accumulation steps × 2,048 tokens = 524,288 tokens per schedule step. Actual valid-row handling, partial batches and skipped optimizer updates mean this product is not proof of the exact number of useful training tokens consumed.

## Training and legacy checkpoint caveats

Stage A used free Kaggle GPU sessions on **one account**, with checkpoints uploaded between sessions. Multi-account quota rotation is not the operating plan. More pretraining depends on available compute; no paid-compute commitment is implied by this README.

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

## Next work

1. Repair the FIM/data-format split and retest plain completion with a fixed, disclosed protocol.
2. Build tool-use SFT from Vision's action registry and nexus-agent schemas, with no FIM on chat/tool data and loss focused on assistant/tool tokens. Evaluate tool selection, argument validity and execution outcomes separately from code benchmarks.
3. Replace the serving placeholder with real inference, then test any JSON constraints and integrations rather than claiming guaranteed tool calls.
4. Improve filtering/deduplication and complete contamination checks before rebuilding data or scaling pretraining.
5. Treat the earlier Stage B pretrain/anneal mix as a proposal. Source access, licenses, retained token counts, compute and evaluation gates need verification before a larger run. Keep the tokenizer compatible unless a separately tested migration is needed.

Reaching 10B tokens is a training milestone, not a promise of quality or the end of development.
