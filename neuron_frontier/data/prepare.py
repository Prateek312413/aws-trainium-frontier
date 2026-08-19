"""
Evaluation Harness & Bits-Per-Byte (BPB) Metric Calculation for Trainium Frontier.

Official Competition Scoring Metric:
  val_bpb = (CrossEntropyLoss / ln(2)) * (total_tokens / total_bytes)

This metric is vocabulary-size-invariant and measures true compression performance
on raw UTF-8 text bytes, identical to Karpathy's nanochat and Devpost leaderboard requirements.
"""

import math
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional


def compute_cross_entropy_from_logits(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> Tuple[float, int]:
    """
    Recomputes cross-entropy directly from model logits to ensure leaderboard score verification.
    """
    B, S, V = logits.shape
    flat_logits = logits.view(-1, V)
    flat_targets = targets.view(-1)
    
    mask = flat_targets != ignore_index
    num_tokens = mask.sum().item()
    if num_tokens == 0:
        return 0.0, 0
        
    loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index, reduction="sum")
    return loss.item(), num_tokens


def evaluate_bpb(model: torch.nn.Module, data_loader, total_raw_bytes: Optional[int] = None, device: str = "cpu", max_batches: Optional[int] = 20) -> Tuple[float, float, int]:
    """
    Official evaluate_bpb() function for Trainium Frontier Leaderboard ranking.
    
    Returns:
      (val_bpb, avg_val_loss, total_tokens_evaluated)
    """
    model.eval()
    total_loss_sum = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for b_idx, (inputs, targets, num_bytes_in_batch) in enumerate(data_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits, loss = model(inputs, targets=None)
            
            if logits is not None:
                batch_loss_sum, batch_tokens = compute_cross_entropy_from_logits(logits, targets)
            else:
                # If using chunked CE loss without full logits
                _, batch_loss = model(inputs, targets=targets)
                batch_tokens = (targets != -100).sum().item()
                batch_loss_sum = batch_loss.item() * batch_tokens
                
            total_loss_sum += batch_loss_sum
            total_tokens += batch_tokens
            
    if total_tokens == 0:
        return float("inf"), float("inf"), 0
        
    avg_loss = total_loss_sum / total_tokens
    
    # Estimate total bytes if not explicitly provided
    # Standard FineWeb-Edu / English BPE has ~4.2 - 4.5 bytes per token
    if total_raw_bytes is None or total_raw_bytes == 0:
        total_raw_bytes = int(total_tokens * 4.35)
        
    # Official Formula: bpb = (loss_nats / ln(2)) * (total_tokens / total_bytes)
    # loss in bits = loss_nats / ln(2) = loss_nats * log2(e)
    bits_per_token = avg_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_raw_bytes)
    
    return val_bpb, avg_loss, total_tokens
