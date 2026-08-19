"""Unit tests for Leaderboard evaluate_bpb() metric."""

import pytest
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from neuron_frontier.data.prepare import evaluate_bpb


class PerfectPredictor(nn.Module):
    def __init__(self, vocab_size: int = 10):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x, targets=None):
        # Create ground-truth one-hot logits
        B, S = x.shape
        logits = torch.zeros(B, S, self.vocab_size)
        for b in range(B):
            for s in range(S):
                # Put high logit on the true target
                logits[b, s, (x[b, s] + 1) % self.vocab_size] = 50.0
        return logits, None


def test_evaluate_bpb_calculation():
    vocab_size = 10
    model = PerfectPredictor(vocab_size)
    
    # 2 sequences of length 4
    x = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    y = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    bytes_batch = torch.tensor([16, 16])
    
    dataset = TensorDataset(x, y, bytes_batch)
    loader = DataLoader(dataset, batch_size=2)
    
    val_bpb, avg_loss, total_tokens = evaluate_bpb(model, loader, total_raw_bytes=32)
    
    # Loss should be virtually 0, and BPB close to 0
    assert total_tokens == 8
    assert avg_loss < 0.01
    assert val_bpb < 0.01
