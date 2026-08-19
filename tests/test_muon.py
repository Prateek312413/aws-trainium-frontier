"""Unit tests for Muon Optimizer and Newton-Schulz iteration."""

import pytest
import torch
import torch.nn as nn
from neuron_frontier.optim.muon import zeropower_via_newtonschulz5, Muon, create_frontier_optimizer


def test_newton_schulz_orthogonalization():
    M = torch.randn(64, 64)
    ortho = zeropower_via_newtonschulz5(M, steps=5)
    
    # Check that ortho @ ortho.T has well-conditioned diagonal and bounded norm
    gram = ortho @ ortho.T
    diag = torch.diag(gram)
    
    # Diagonals are strictly bounded between 0.6 and 1.2
    assert diag.min().item() > 0.6
    assert diag.max().item() < 1.2
    assert abs(diag.mean().item() - 0.85) < 0.25


def test_muon_step():
    lin = nn.Linear(32, 32, bias=False)
    opt = Muon(lin.parameters(), lr=0.01)
    
    x = torch.randn(4, 32)
    loss = (lin(x) ** 2).sum()
    loss.backward()
    
    w_before = lin.weight.clone()
    opt.step()
    w_after = lin.weight.clone()
    
    # Ensure weights were updated
    assert not torch.allclose(w_before, w_after)


def test_dual_optimizer_creation():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_embeddings = nn.Embedding(100, 32)
            self.linear = nn.Linear(32, 32, bias=False)
            self.norm = nn.LayerNorm(32)
            self.lm_head = nn.Linear(32, 100, bias=False)
            
    model = DummyModel()
    muon_opt, adamw_opt = create_frontier_optimizer(model)
    
    assert len(muon_opt.param_groups[0]["params"]) == 1  # only self.linear.weight
    assert len(adamw_opt.param_groups[0]["params"]) == 4  # tok_emb, norm.w, norm.b, lm_head.w
