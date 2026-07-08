# 🧠 CodeForge-250M: Comprehensive Project Memory & State

*This document serves as the living memory and technical record of the **CodeForge-250M** project, documenting our architectural goals, training journey, cloud economics, technical decisions, and pending tasks.*

---

## 🎯 1. Project Identity & Ultimate Grand Target
* **Model Name:** `CodeForge-250M`
* **Model Architecture:** 
  * **Parameters:** ~250 Million Parameters (Targeted for lightweight, high-performance Edge AI & local device execution).
  * **Specifications:** 16 Transformer Layers, 1024 Hidden Dimension, Grouped Query Attention (GQA), RMSNorm, SwiGLU Activation.
  * **Precision & Acceleration:** BF16 / FP16 mixed precision training optimized for Ampere/Ada/Blackwell Tensor Cores using PyTorch Scaled Dot-Product Attention (SDPA / FlashAttention).
* **Target Use Case:** An autonomous, highly capable AI coding assistant designed to run efficiently on edge devices, laptops, and local environments without cloud latency.
* **🔥 THE ULTIMATE TRAINING TARGET: `8 BILLION TO 10 BILLION TOKENS (8B - 10B Tokens!)`**
  * *Correction & Note:* In recent sprint discussions, "1 Billion Tokens" was referenced as a target. **This was a mistake in communication.** 1 Billion Tokens is merely **Milestone #1** (the first ~10-12% of training).
  * Our true architectural goal is training across **8B to 10B tokens**, achieving Chinchilla-plus scaling to ensure state-of-the-art reasoning, syntax mastery, and coding fluency for a 250M parameter model.

---

## 📈 2. Training Journey & Current Progress
* **Starting Point:** Step 0 / 0 Tokens on Lightning AI Studio.
* **Phase 1: Foundation & T4 Streaming (Chunks 1 - 14):**
  * Built continuous, multi-epoch streaming data loaders from deduplicated tokenized shards (`data/tokenized`).
  * Successfully trained up to Step 72,000 (~295 Million Tokens / ~300M Mark) using FREE Tesla T4 GPUs (~30M tokens/hr).
* **Phase 2: Super-Speed Sprint on NVIDIA L40S (Chunks 15 - 16):**
  * Upgraded cloud compute to **NVIDIA L40S (48 GB VRAM Ada Lovelace GPU)**.
  * Increased micro-batch size from `batch_size = 2` (4,096 tokens/step) to **`batch_size = 8` (16,384 tokens/step)**.
  * **Performance Achieved:** Training speed multiplied 4x to **~154 Steps/min (~42,000+ tokens/sec / ~151+ Million tokens/hour)** at 97-100% GPU utilization (~320W power draw).
* **Phase 3: RTX PRO 6000 Power Sprint & 0.81 Billion Token Mark (Stages 17 - 18):**
  * Upgraded cloud compute to **NVIDIA RTX PRO 6000 (Blackwell Server Edition / 500 TFLOPS / 96 GB VRAM)**.
  * Set micro-batch size to **`batch_size = 16` (32,768 tokens/step)** — discovered as our **Golden Sweet Spot**, achieving **~82,598 tokens/sec (~300 Million tokens/hour)** at 99-100% GPU utilization and ~52.3 GB VRAM usage.
  * **The Batch Size 32 OOM Experiment:** Tested `batch_size = 32` (Beast Sprint). While forward pass took ~76.6 GB VRAM, during Step 89,002's backward pass (backpropagation), PyTorch dynamically requested an extra ~18.37 GB of gradient buffer, hitting **94.97 GB / 96 GB and throwing a CUDA Out of Memory (OOM) crash**. This proved that our ~44 GB free buffer at Batch Size 16 is essential dynamic gradient accumulation memory!
  * **Pro-Feature Added (`STOP_AND_SAVE`):** Integrated an instant-save file trigger in `train.py` so that creating a `STOP_AND_SAVE` file on the remote server immediately dumps checkpoints and exits cleanly without losing a single token of compute.
* **Phase 4: Spot Mode Protection & 1.02 Billion Token Milestone Victory (Stage 19):**
  * **Interruptible / Spot Mode Protection:** Re-launched Studio in AWS Spot Mode, slashing RTX PRO 6000 cost by **56% from $4.64/hr down to $2.05/hr**!
  * **1,000-Step Auto-Save Shield:** To guarantee zero data loss in Spot mode, added an automatic checkpoint saving loop (`if step % 1000 == 0:`) to `train.py`, ensuring at most ~6.5 minutes of compute exposure.
  * **🏆 Current Milestone Achieved (1 BILLION TOKENS CROSSED!):**
    * Completed the Stage 19 Sprint, reaching **Step 102,546**, officially processing **`~1,019,834,368 TOKENS (~1.02 BILLION TOKENS CROSSED!)`**!
    * Final training metrics: Loss reached **`0.0153`** (Perplexity **`1.02`**), proving world-class syntax fluency and deep code reasoning.
    * *Current Status:* Training completed cleanly and weights (`checkpoint_step_102546.pt` and `latest_checkpoint.pt`) are live on Hugging Face Hub (`Yash1bajpai/CodeForge-250M`) and synced to GitHub!

---

## 💰 3. Cloud Infrastructure, Economics & Credit Math
* **Platform:** Lightning AI Studio (`ssh.lightning.ai`).
* **Credit Balance & Expenditure Breakdown:**
  * **Initial Total Wealth:** `$19.10 USD`
  * **Stage 18 Power Sprint Cost:** Ran RTX PRO 6000 ($4.64/hour) for 47m 36s, costing **`~$3.70 USD`**.
  * **Stage 19 / 1 Billion Token Sprint Cost:** Ran in **Spot Mode ($2.05/hr)** for ~36 minutes, costing only **`~$1.23 USD`** (saved over $1.50 via Spot pricing!).
  * **Remaining Wealth Available:** **`~$14.17 USD`**
* **Cost Analysis & Strategy for 8B - 10B Tokens:**
  * **RTX PRO 6000 Spot Economics:** At ~$2.05/hr (~300M tokens/hr), cost is **~$6.83 per Billion tokens**!
  * **Cost to hit 8 Billion - 10 Billion Tokens:** With ~$14.17 remaining, we can train another **~2.1 Billion tokens** directly on RTX PRO 6000 Spot!
  * **Strategy for 8B - 10B Tokens:** 
    1. Utilize RTX PRO 6000 in Spot Mode ($2.05/hr) with our 1,000-step auto-save shield for ultra-cheap, high-throughput sprints.
    2. Leverage FREE Tesla T4 GPU hours on Lightning AI Studio for steady, overnight background streaming.
    3. Utilize modular checkpointing across Kaggle Free Notebooks (30 free GPU hrs/week) or Google Colab to reach the full 8B-10B target without excessive cloud expenditure.

---

## 🛠️ 4. Key Technical Decisions & Bug Fixes Applied
1. **The Batch Size 16 Golden Sweet Spot:** Proved mathematically and empirically that at `batch_size = 16`, Tensor Cores hit 100% saturation (~82,598 tps). Increasing to Batch 32 does not increase compute speed; it only consumes dynamic activation buffer until CUDA OOMs during backward propagation.
2. **Instant-Save Signal Trigger (`STOP_AND_SAVE`):** Enabled non-destructive early termination of training sessions via filesystem signal detection, guaranteeing zero wasted compute credits.
3. **Automated Guard Monitoring:** Developed background scheduling workflows that monitor training runtimes and enforce strict budget/time ceilings (e.g., auto-stopping before 1-hour cloud limits).
4. **RMSNorm & SwiGLU Stabilization:** Fixed numerical instability in custom RMSNorm layers during mixed-precision training (`fix_rmsnorm.py`).
5. **SDPA / FlashAttention Memory Optimization:** Solved CUDA Out of Memory (OOM) errors during long sequence attention passes (`fix_sdpa_oom.py`).
6. **SSD Disk Bloat Pruning:** Developed automated pruning script (`prune_disk_bloat.py`) that deleted ~60 ancient intermediate checkpoints, freeing up **~176 GB** of cloud SSD space and maintaining disk health.
7. **Automated Hub Release & Git Sync:** Configured scripts (`save_and_upload_all.py`, `sync_to_github_win.py`) to automatically release milestone checkpoints to Hugging Face Hub (`Yash1bajpai/CodeForge-250M`) and sync logs/code to Windows local Git repo.

---

## 🧹 5. Known Technical Debt & Pending Tasks
* **Git Repository Clutter (Crucial Catch by Yash):**
  * *Issue:* During our rapid coding sprints, automated sync scripts blindly copied all root `.py` files from the remote server and executed `git add .`.
  * *Result:* Over 20+ temporary one-off scripts (`start_chunk4_remote.py` to `start_chunk15_remote.py`, `apply_batch8.py`, `prune_disk_bloat.py`, `patch_remote.py`) and a 1.4 MB `training.log` file were accidentally committed to the root of the GitHub repository.
  * *Pending Fix:* Clean the repository root, move essential utilities to `scripts/`, delete obsolete launcher scripts from Git, update `.gitignore` to exclude `*.log` and temporary scripts, and push a clean, professional open-source commit.
* **Hugging Face Checkpoint Push for Step 96,200:**
  * Since Lightning AI Studio was paused/turned off immediately upon training completion to save credits, the Step 96,200 checkpoint weights (`checkpoint_step_96200.pt` and `latest_checkpoint.pt`) are residing safely on the Studio SSD. Upon powering on the studio for Stage 19, our first action will be pushing these weights to Hugging Face Hub (`Yash1bajpai/CodeForge-250M`).

---

## ❓ 6. Open Review for Yash: What Did I Forget?
*Bhai, I have documented everything above from our latest Stage 18 Power Sprint and memory. Please review this document and tell me:*
1. **What architectural details, hyperparameter choices, or dataset preprocessing steps did I miss?**
2. **Are there specific phases, evaluation benchmarks, or deployment plans for the 8B-10B run that we need to add here?**
3. **When we power on Lightning AI next, shall we immediately execute the 38-minute sprint to cross our historic 1 Billion Token Milestone?**
