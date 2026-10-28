import math
import torch
from typing import Dict

def get_lr_cosine_schedule(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """
    Cosine Learning Rate Decay with Linear Warmup.
    Warmup: linearly increase from 0 to max_lr over warmup_steps.
    Decay: cosine curve from max_lr down to min_lr (10% of max_lr) over remaining steps.
    """
    if step < warmup_steps:
        return max_lr * (step / max(1, warmup_steps))
    if step >= max_steps:
        return min_lr
    
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def calculate_flops_per_step(model, batch_size_seqs: int, seq_len: int) -> float:
    """
    Estimates hardware FLOPs per training step using standard transformer approximation:
    FLOPs = 6 * N_params * batch_size * seq_len
    """
    params = sum(p.numel() for p in model.parameters())
    return 6.0 * params * batch_size_seqs * seq_len


def val_trend_diverged(history, half=5, margin=0.05):
    """True when validation loss is clearly rising, not just noisy.

    Compares the mean of the latest `half` evaluations with the mean of the
    `half` before them. Needs 2*half points and a rise above `margin`, so a
    few tiny single-batch fluctuations (4 strictly rising points) cannot stop
    a run whose loss is still falling overall.
    """
    if len(history) < 2 * half:
        return False
    recent = sum(history[-half:]) / half
    previous = sum(history[-2 * half:-half]) / half
    if recent != recent or previous != previous:
        return True  # NaN validation loss is never healthy
    return recent > previous + margin
