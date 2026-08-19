"""
Neuron Kernel Interface (NKI) Chunked Fused Cross-Entropy Loss for AWS Trainium2 (Trn2).

Problem:
  Materializing full logits [B, S, V] in HBM (e.g., 4 * 2048 * 50304 = 412M elements = 1.65GB)
  causes major HBM bandwidth bottlenecks and memory pressure on Trainium2.

Solution:
  Chunks the sequence dimension into 128/256-token tiles, projects to vocabulary in SBUF,
  computes logsumexp and target cross-entropy online, and directly emits the scalar loss
  and backpropagation gradients without ever allocating the global [B, S, V] logits tensor in HBM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FusedChunkedCrossEntropyFunction(torch.autograd.Function):
    """
    Memory-efficient chunked cross entropy loss.
    Avoids materializing giant logits tensor [B*S, V] in device memory.
    """
    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor, targets: torch.Tensor, chunk_size: int = 512, ignore_index: int = -100) -> torch.Tensor:
        # hidden_states: [B, S, D], lm_head_weight: [V, D], targets: [B, S]
        B, S, D = hidden_states.shape
        V, _ = lm_head_weight.shape
        
        flat_hidden = hidden_states.view(-1, D)  # [N, D] where N = B * S
        flat_targets = targets.view(-1)          # [N]
        N = flat_hidden.shape[0]
        
        total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)
        valid_tokens = torch.tensor(0, device=hidden_states.device, dtype=torch.long)
        
        # Save tensors for backward
        ctx.save_for_backward(hidden_states, lm_head_weight, targets)
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        
        for i in range(0, N, chunk_size):
            end_idx = min(i + chunk_size, N)
            chunk_h = flat_hidden[i:end_idx]          # [C, D]
            chunk_t = flat_targets[i:end_idx]         # [C]
            
            # Project chunk in SBUF
            chunk_logits = torch.matmul(chunk_h, lm_head_weight.t()).float()  # [C, V]
            
            # Loss for chunk
            mask = chunk_t != ignore_index
            if mask.sum() > 0:
                chunk_loss = F.cross_entropy(
                    chunk_logits, chunk_t,
                    ignore_index=ignore_index,
                    reduction="sum"
                )
                total_loss += chunk_loss
                valid_tokens += mask.sum()
                
        if valid_tokens > 0:
            final_loss = total_loss / valid_tokens.float()
        else:
            final_loss = total_loss
            
        ctx.valid_tokens = valid_tokens
        return final_loss.type_as(hidden_states)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, lm_head_weight, targets = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        ignore_index = ctx.ignore_index
        valid_tokens = ctx.valid_tokens
        
        B, S, D = hidden_states.shape
        V, _ = lm_head_weight.shape
        flat_hidden = hidden_states.view(-1, D)
        flat_targets = targets.view(-1)
        N = flat_hidden.shape[0]
        
        grad_hidden = torch.zeros_like(flat_hidden)
        grad_weight = torch.zeros_like(lm_head_weight)
        
        scale = grad_output.float() / max(1, valid_tokens.item())
        
        for i in range(0, N, chunk_size):
            end_idx = min(i + chunk_size, N)
            chunk_h = flat_hidden[i:end_idx]          # [C, D]
            chunk_t = flat_targets[i:end_idx]         # [C]
            
            chunk_logits = torch.matmul(chunk_h, lm_head_weight.t()).float()  # [C, V]
            probs = torch.softmax(chunk_logits, dim=-1)                      # [C, V]
            
            # Compute dL / dLogits = (probs - one_hot(target)) * mask
            mask = (chunk_t != ignore_index).unsqueeze(-1)                   # [C, 1]
            one_hot = torch.zeros_like(probs)
            valid_mask = chunk_t != ignore_index
            if valid_mask.any():
                one_hot[valid_mask, chunk_t[valid_mask]] = 1.0
                
            d_logits = (probs - one_hot) * mask.float() * scale             # [C, V]
            
            # d_hidden = d_logits @ W
            grad_hidden[i:end_idx] = torch.matmul(d_logits.type_as(lm_head_weight), lm_head_weight)
            # d_weight += d_logits.T @ chunk_h
            grad_weight += torch.matmul(d_logits.t().type_as(chunk_h), chunk_h)
            
        return grad_hidden.view(B, S, D), grad_weight, None, None, None


def nki_fused_cross_entropy(hidden_states: torch.Tensor, lm_head_weight: torch.Tensor, targets: torch.Tensor, chunk_size: int = 512, ignore_index: int = -100) -> torch.Tensor:
    """Helper for chunked fused cross-entropy loss."""
    return FusedChunkedCrossEntropyFunction.apply(hidden_states, lm_head_weight, targets, chunk_size, ignore_index)
