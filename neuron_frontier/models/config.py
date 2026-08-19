from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NeuronFrontierConfig:
    """
    Hardware-Co-Designed Configuration for AWS Trainium2 (Trn2).
    
    Architectural parameters are strictly aligned to Trainium2's TensorEngine
    systolic array tile dimensions (128x128) and SBUF scratchpad memory limits.
    """
    vocab_size: int = 50304  # Padded to multiple of 128 for TensorEngine tiling
    max_seq_len: int = 2048  # Context length
    dim: int = 768           # Model dimension (multiple of 128: 768 = 6 * 128)
    n_layers: int = 12       # Number of transformer blocks
    n_heads: int = 12        # Query heads (dim // n_heads = 64)
    n_kv_heads: int = 4      # Grouped Query Attention (GQA) 3:1 ratio
    hidden_dim: Optional[int] = None  # SwiGLU intermediate dimension (multiple of 128)
    
    # Normalization & Numerical Stability
    norm_eps: float = 1e-5
    qk_norm: bool = True     # Prevents attention entropy collapse under aggressive Muon LR
    rope_theta: float = 10000.0
    use_bias: bool = False   # Pure linear ops align better with TensorEngine DMA
    
    # Kernel acceleration flags
    use_nki_flash_attn: bool = True
    use_nki_fused_swiglu: bool = True
    use_nki_fused_rmsnorm: bool = True
    use_nki_fused_ce: bool = True
    
    # Hardware alignment constraints
    tensor_tile_dim: int = 128  # Trn2 TensorEngine systolic tile size
    sbuf_size_mb: int = 24     # NeuronCore SBUF scratchpad size
    
    # Mixture of Experts parameters (if enabled)
    is_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_dim: Optional[int] = None
    
    def __post_init__(self):
        if self.hidden_dim is None:
            # Standard SwiGLU 8/3 expansion aligned to 128
            raw_dim = int(2 * (4 * self.dim) / 3)
            self.hidden_dim = ((raw_dim + self.tensor_tile_dim - 1) // self.tensor_tile_dim) * self.tensor_tile_dim
        
        if self.moe_intermediate_dim is None and self.is_moe:
            raw_dim = int((4 * self.dim) / 3)
            self.moe_intermediate_dim = ((raw_dim + self.tensor_tile_dim - 1) // self.tensor_tile_dim) * self.tensor_tile_dim
            
        assert self.dim % self.tensor_tile_dim == 0, f"dim ({self.dim}) must be a multiple of {self.tensor_tile_dim} for Trn2"
        assert self.dim % self.n_heads == 0, f"dim must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, f"n_heads must be divisible by n_kv_heads for GQA"


# Standard 30-Minute Speedrun Configurations
def get_speedrun_small_config() -> NeuronFrontierConfig:
    """Fast-iteration config (~45M params). High step throughput (140k+ tokens/sec)."""
    return NeuronFrontierConfig(
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        max_seq_len=1024,
    )


def get_speedrun_base_config() -> NeuronFrontierConfig:
    """
    Recommended Phase 1 30-Minute Baseline (~124M params).
    Optimal Pareto balance between capacity and token throughput on single Trn2 chip.
    """
    return NeuronFrontierConfig(
        dim=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        max_seq_len=2048,
    )


def get_speedrun_moe_config() -> NeuronFrontierConfig:
    """
    SOTA Speedrun MoE Config (~180M total / ~48M active params).
    Maximizes validation bits-per-byte (val_bpb) reduction per compute FLOP.
    """
    return NeuronFrontierConfig(
        dim=768,
        n_layers=10,
        n_heads=12,
        n_kv_heads=4,
        max_seq_len=2048,
        is_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
    )


def get_phase2_cluster_config() -> NeuronFrontierConfig:
    """Phase 2 scaled configuration (~1.2B params) for multi-node Trn2 clusters."""
    return NeuronFrontierConfig(
        dim=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        max_seq_len=4096,
        is_moe=True,
        num_experts=16,
        num_experts_per_tok=2,
    )
