"""
Learning Rate Schedules for 30-Minute Wall-Clock Speedruns on AWS Trainium2.

Includes:
1. Warmup-Stable-Decay (WSD): Maximizes token learning rate efficiency over a fixed time budget.
2. Cosine Decay with Warmup and configurable minimum learning rate floor.
"""

import math


def get_lr_wsd(step: int, total_steps: int, warmup_steps: int, decay_steps: int, max_lr: float, min_lr: float = 0.0) -> float:
    """
    Warmup-Stable-Decay (WSD) Schedule.
    1. Linear warmup from 0 to max_lr for `warmup_steps`.
    2. Constant plateau at `max_lr` until `total_steps - decay_steps`.
    3. Cosine decay down to `min_lr` during the final `decay_steps`.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    
    decay_start = total_steps - decay_steps
    if step < decay_start:
        return max_lr
    
    # Annealing phase
    decay_progress = (step - decay_start) / max(1, decay_steps)
    decay_progress = min(1.0, max(0.0, decay_progress))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr + coeff * (max_lr - min_lr)


def get_lr_cosine(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float = 0.0) -> float:
    """Cosine Annealing with Linear Warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)
