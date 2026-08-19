"""
Hardware-Co-Designed Transformer Architecture for AWS Trainium2.

Architectural Innovations:
1. TensorEngine 128x128 Tile Alignment: Hidden states, projections, and heads align strictly with Trainium2 tiles.
2. QK-Norm: Applies RMSNorm to Queries and Keys prior to attention dot-product to prevent entropy collapse during aggressive Muon training.
3. Grouped Query Attention (GQA): Reduces KV cache footprint in SBUF while maintaining expressive query capability.
4. Fused SwiGLU & RMSNorm: Interleaved NKI custom kernels for maximum throughput.
5. Dynamic RoPE: High-frequency rotary embeddings with float32 precision accumulator.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from neuron_frontier.models.config import NeuronFrontierConfig
from neuron_frontier.kernels.nki_fused_rmsnorm import NKIRMSNorm
from neuron_frontier.kernels.nki_fused_swiglu import nki_fused_swiglu
from neuron_frontier.kernels.nki_flash_attn import nki_flash_attention
from neuron_frontier.kernels.nki_fused_cross_entropy import nki_fused_cross_entropy


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE) optimized for Trn2 VectorEngine."""
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, H, S, D]
    # cos, sin: [S, D] -> [1, 1, S, D]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class Trn2Attention(nn.Module):
    """
    Hardware-Aligned Grouped Query Attention with QK-Norm and NKI FlashAttention.
    """
    def __init__(self, config: NeuronFrontierConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.num_kv_groups = config.n_heads // config.n_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=config.use_bias)
        
        # QK-Norm layers (vital for Muon optimizer stability)
        if config.qk_norm:
            self.q_norm = NKIRMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = NKIRMSNorm(self.head_dim, eps=config.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H_q, S, D_h]
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, H_kv, S, D_h]
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, H_kv, S, D_h]
        
        # Apply QK-Norm
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
            
        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Repeat KV heads for GQA if needed
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
            
        # NKI FlashAttention
        attn_out = nki_flash_attention(q, k, v, is_causal=True)  # [B, H, S, D_h]
        
        # Reshape and project out
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(attn_out)


class Trn2MLP(nn.Module):
    """
    Hardware-Aligned SwiGLU MLP for Trainium2 TensorEngine.
    """
    def __init__(self, config: NeuronFrontierConfig):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.dim, config.hidden_dim, bias=config.use_bias)
        self.up_proj = nn.Linear(config.dim, config.hidden_dim, bias=config.use_bias)
        self.down_proj = nn.Linear(config.hidden_dim, config.dim, bias=config.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        if self.config.use_nki_fused_swiglu:
            act = nki_fused_swiglu(gate, up)
        else:
            act = F.silu(gate) * up
            
        return self.down_proj(act)


class Trn2TransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer Block with NKI acceleration."""
    def __init__(self, config: NeuronFrontierConfig):
        super().__init__()
        self.attn_norm = NKIRMSNorm(config.dim, eps=config.norm_eps)
        self.attn = Trn2Attention(config)
        self.mlp_norm = NKIRMSNorm(config.dim, eps=config.norm_eps)
        self.mlp = Trn2MLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connection
        h = x + self.attn(self.attn_norm(x), cos, sin)
        out = h + self.mlp(self.mlp_norm(h))
        return out


class Trn2TransformerLM(nn.Module):
    """
    Complete Hardware-Co-Designed Language Model for AWS Trainium2.
    """
    def __init__(self, config: NeuronFrontierConfig):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.rope = RotaryEmbedding(config.dim // config.n_heads, max_seq_len=config.max_seq_len, theta=config.rope_theta)
        
        self.layers = nn.ModuleList([
            Trn2TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.final_norm = NKIRMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight tying option (untied by default for better bpb scaling)
        self._init_weights()

    def _init_weights(self):
        # Small standard deviation initialization for stable Muon training
        std = 0.02
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=std)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)
        
        for name, param in self.named_parameters():
            if "out_proj.weight" in name or "down_proj.weight" in name:
                # Scale residual output projections by 1 / sqrt(2 * n_layers)
                nn.init.normal_(param, mean=0.0, std=std / math.sqrt(2 * self.config.n_layers))
            elif "proj.weight" in name:
                nn.init.normal_(param, mean=0.0, std=std)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S = input_ids.shape
        x = self.tok_embeddings(input_ids)
        
        cos, sin = self.rope(x, S)
        
        for layer in self.layers:
            x = layer(x, cos, sin)
            
        x = self.final_norm(x)
        
        loss = None
        if targets is not None:
            if self.config.use_nki_fused_ce:
                loss = nki_fused_cross_entropy(x, self.lm_head.weight, targets, chunk_size=512)
                logits = None  # Saves memory
            else:
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1), ignore_index=-100)
        else:
            logits = self.lm_head(x)
            
        return logits, loss

    def get_num_params(self, exclude_embeddings: bool = False) -> int:
        num = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            num -= self.tok_embeddings.weight.numel()
        return num
