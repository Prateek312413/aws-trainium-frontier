"""Unit tests for NKI Custom Hardware Kernels."""

import pytest
import math
import torch
import torch.nn.functional as F

from neuron_frontier.kernels.nki_flash_attn import nki_flash_attention
from neuron_frontier.kernels.nki_fused_swiglu import nki_fused_swiglu
from neuron_frontier.kernels.nki_fused_rmsnorm import NKIRMSNorm
from neuron_frontier.kernels.nki_fused_cross_entropy import nki_fused_cross_entropy


def test_flash_attention_parity():
    B, H, S, D = 2, 4, 64, 32
    q = torch.randn(B, H, S, D, requires_grad=True)
    k = torch.randn(B, H, S, D, requires_grad=True)
    v = torch.randn(B, H, S, D, requires_grad=True)
    
    # Reference
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    mask = torch.triu(torch.full((S, S), float('-inf')), diagonal=1)
    ref_out = torch.matmul(F.softmax(scores + mask, dim=-1), v)
    
    # NKI Flash Attention
    nki_out = nki_flash_attention(q, k, v, is_causal=True)
    
    assert torch.allclose(ref_out, nki_out, atol=1e-4)


def test_fused_swiglu_parity():
    N, D = 128, 256
    gate = torch.randn(N, D, requires_grad=True)
    up = torch.randn(N, D, requires_grad=True)
    
    ref_out = F.silu(gate) * up
    nki_out = nki_fused_swiglu(gate, up)
    
    assert torch.allclose(ref_out, nki_out, atol=1e-5)
    
    # Backward test
    ref_out.sum().backward()
    g_gate_ref = gate.grad.clone()
    g_up_ref = up.grad.clone()
    
    gate.grad.zero_()
    up.grad.zero_()
    nki_out.sum().backward()
    
    assert torch.allclose(g_gate_ref, gate.grad, atol=1e-5)
    assert torch.allclose(g_up_ref, up.grad, atol=1e-5)


def test_fused_rmsnorm_parity():
    N, D = 128, 256
    x = torch.randn(N, D, requires_grad=True)
    norm = NKIRMSNorm(D)
    
    out = norm(x)
    assert out.shape == (N, D)
    
    # Check variance normalization
    var = torch.mean(out * out, dim=-1)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-2)


def test_fused_cross_entropy_parity():
    B, S, D, V = 2, 32, 128, 512
    h = torch.randn(B, S, D, requires_grad=True)
    w = torch.randn(V, D, requires_grad=True)
    targets = torch.randint(0, V, (B, S))
    
    # Reference
    logits = torch.matmul(h, w.t())
    ref_loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
    
    # NKI Chunked Fused CE
    nki_loss = nki_fused_cross_entropy(h, w, targets, chunk_size=16)
    
    assert torch.allclose(ref_loss, nki_loss, atol=1e-4)
