"""
Muon (Momentum Orthogonalized by Newton-Schulz) Optimizer for AWS Trainium2.

Muon applies Newton-Schulz orthogonalization to momentum buffers of 2D weight matrices,
keeping spectral norms strictly bounded and enabling up to 2-3x faster convergence
than standard AdamW within the 30-minute compute budget.

1D tensors (RMSNorm gains, biases) and Embedding/LM-head weights are optimized via AdamW.
"""

import math
import torch
from typing import List, Dict, Any, Tuple, Optional


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz 5th-order iteration for matrix orthogonalization.
    Computes (G @ G.T)^(-1/2) @ G for spectral normalization.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
    
    # Scale by Frobenius norm to ensure top singular value <= 1
    X = X / (X.norm() + eps)
    
    # Transpose if tall matrix to ensure inner dimension is smaller
    if G.size(0) > G.size(1):
        X = X.T
        
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        
    if G.size(0) > G.size(1):
        X = X.T
        
    return X.type_as(G)


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer for 2D matrix weights.
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                
                # Apply decoupled weight decay
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                    
                state = self.state[p]
                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = torch.zeros_like(p)
                else:
                    buf = state["momentum_buffer"]
                    
                # Update momentum
                buf.mul_(momentum).add_(g)
                
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf
                    
                # Orthogonalize via Newton-Schulz
                if p.ndim >= 2:
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    # Scale update by aspect ratio: sqrt(max(1, d_out / d_in))
                    aspect = math.sqrt(max(1.0, p.size(0) / p.size(1)))
                    update.mul_(aspect)
                    
                p.add_(update, alpha=-lr)
                
        return loss


def create_frontier_optimizer(model: torch.nn.Module, muon_lr: float = 0.02, adamw_lr: float = 0.001, weight_decay: float = 0.01, betas: Tuple[float, float] = (0.9, 0.95)) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """
    Creates the dual Muon + AdamW optimizer pair as recommended for Trainium Frontier speedruns.
    - Muon: optimizes 2D linear weight matrices.
    - AdamW: optimizes 1D tensors (RMSNorm gains, biases) and Embedding / LM-Head layers.
    """
    muon_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Check if 2D matrix in transformer body
        if param.ndim == 2 and "tok_embeddings" not in name and "lm_head" not in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)
            
    muon_opt = Muon(muon_params, lr=muon_lr, momentum=0.95, weight_decay=weight_decay)
    adamw_opt = torch.optim.AdamW(adamw_params, lr=adamw_lr, betas=betas, weight_decay=weight_decay, eps=1e-8)
    
    return muon_opt, adamw_opt
