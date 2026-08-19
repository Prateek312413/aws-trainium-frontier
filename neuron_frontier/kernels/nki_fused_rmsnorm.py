"""
Neuron Kernel Interface (NKI) Fused RMSNorm for AWS Trainium2 (Trn2).

Computes Root Mean Square Layer Normalization:
  RMS(x) = sqrt( 1/d * sum(x_i^2) + eps )
  y = (x / RMS(x)) * weight

In NKI, variance reduction and weight scaling occur directly in SBUF registers,
minimizing memory traffic.
"""

import torch
import torch.nn as nn
from neuron_frontier.kernels.nki_simulator import is_nki_available, trn2_tile_pad, trn2_tile_unpad

if is_nki_available():
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa

    @nki.jit
    def nki_rmsnorm_fwd_kernel(x_ref, weight_ref, out_ref, eps=1e-5):
        """Native Trainium2 SBUF Fused RMSNorm Forward."""
        batch, seq_len, dim = x_ref.shape
        tile_s = 128
        n_tiles_s = seq_len // tile_s
        
        # Load weight into SBUF once per core
        w_tile = nl.ndarray((1, dim), dtype=weight_ref.dtype, buffer=nl.sbuf)
        nisa.tensor_load(w_tile, weight_ref)
        
        for b in nl.affine_range(batch):
            for t in nl.affine_range(n_tiles_s):
                x_tile = nl.ndarray((tile_s, dim), dtype=x_ref.dtype, buffer=nl.sbuf)
                nisa.tensor_load(x_tile, x_ref[b, t * tile_s:(t + 1) * tile_s, :])
                
                # Compute variance in SBUF: mean(x^2)
                x_f32 = x_tile.astype(nl.float32)
                var = nl.mean(x_f32 * x_f32, axis=-1, keepdims=True)
                rsqrt = 1.0 / nl.sqrt(var + eps)
                
                # Normalize and scale
                normed = x_f32 * rsqrt
                out_tile = normed * w_tile.astype(nl.float32)
                
                nisa.tensor_store(out_ref[b, t * tile_s:(t + 1) * tile_s, :], out_tile.astype(out_ref.dtype))


class NKIRMSNormFunction(torch.autograd.Function):
    """Autograd Function for Fused RMSNorm with analytical backward gradients."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        ctx.eps = eps
        
        # Compute RMSNorm
        x_f32 = x.float()
        var = torch.mean(x_f32 * x_f32, dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(var + eps)
        normed = x_f32 * rsqrt
        out = (normed * weight).type_as(x)
        
        ctx.save_for_backward(x, weight, rsqrt)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight, rsqrt = ctx.saved_tensors
        eps = ctx.eps
        dim = x.shape[-1]
        
        x_f32 = x.float()
        grad_out_f32 = grad_out.float()
        
        # d_weight = sum(grad_out * normed)
        normed = x_f32 * rsqrt
        grad_weight = torch.sum(grad_out_f32 * normed, dim=(0, 1))
        
        # d_x = (grad_out * weight - normed * mean(grad_out * weight * normed)) * rsqrt
        gw = grad_out_f32 * weight
        mean_gw_normed = torch.mean(gw * normed, dim=-1, keepdim=True)
        grad_x = (gw - normed * mean_gw_normed) * rsqrt
        
        return grad_x.type_as(x), grad_weight.type_as(weight), None


class NKIRMSNorm(nn.Module):
    """Drop-in Fused RMSNorm layer for Trainium2."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return NKIRMSNormFunction.apply(x, self.weight, self.eps)
