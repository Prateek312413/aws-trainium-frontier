"""Unit tests for Trn2 Transformer and Sparse MoE models."""

import pytest
import torch
from neuron_frontier.models.config import NeuronFrontierConfig
from neuron_frontier.models.trn2_transformer import Trn2TransformerLM
from neuron_frontier.models.trn2_moe import Trn2MoELM


def test_transformer_lm_forward():
    config = NeuronFrontierConfig(
        vocab_size=1024,
        dim=256,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
        use_nki_fused_ce=False
    )
    model = Trn2TransformerLM(config)
    
    input_ids = torch.randint(0, 1024, (2, 64))
    targets = torch.randint(0, 1024, (2, 64))
    
    logits, loss = model(input_ids, targets=targets)
    
    assert logits.shape == (2, 64, 1024)
    assert loss is not None
    assert loss.item() > 0
    
    # Backward pass
    loss.backward()
    assert model.tok_embeddings.weight.grad is not None


def test_moe_lm_forward():
    config = NeuronFrontierConfig(
        vocab_size=1024,
        dim=256,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
        is_moe=True,
        num_experts=4,
        num_experts_per_tok=2,
        use_nki_fused_ce=False
    )
    model = Trn2MoELM(config)
    
    input_ids = torch.randint(0, 1024, (2, 64))
    targets = torch.randint(0, 1024, (2, 64))
    
    logits, loss = model(input_ids, targets=targets)
    
    assert logits.shape == (2, 64, 1024)
    assert loss is not None
    assert loss.item() > 0
    
    loss.backward()
    assert model.tok_embeddings.weight.grad is not None


def test_hardware_tile_alignment():
    config = NeuronFrontierConfig(
        vocab_size=50304,
        dim=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4
    )
    # Check 128 tile multiples
    assert config.dim % 128 == 0
    assert config.hidden_dim % 128 == 0
    assert config.vocab_size % 128 == 0
