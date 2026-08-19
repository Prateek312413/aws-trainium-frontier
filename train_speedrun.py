"""
AWS Trainium Frontier: 30-Minute Wall-Clock Speedrun Training Loop.

This is the main entrypoint submitted for Phase 1 leaderboard evaluation.
Enforces an exact 30.00-minute compute budget on a single Trainium2 chip,
tracking validation bits-per-byte (val_bpb), throughput (tok/sec), and MFU.
"""

import os
import sys
import time
import math
import argparse
import logging
import torch
import torch.nn as nn

from neuron_frontier.models.config import (
    get_speedrun_small_config,
    get_speedrun_base_config,
    get_speedrun_moe_config,
    NeuronFrontierConfig
)
from neuron_frontier.models.trn2_transformer import Trn2TransformerLM
from neuron_frontier.models.trn2_moe import Trn2MoELM
from neuron_frontier.optim.muon import create_frontier_optimizer
from neuron_frontier.optim.schedules import get_lr_wsd, get_lr_cosine
from neuron_frontier.data.dataset import create_speedrun_dataloaders
from neuron_frontier.data.prepare import evaluate_bpb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TrainiumFrontier")


def get_device() -> torch.device:
    """Auto-detects Trainium2 XLA device or fallback to CUDA/CPU."""
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        logger.info(f"Detected AWS NeuronCore XLA Device: {device}")
        return device
    except (ImportError, ModuleNotFoundError):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Running on NVIDIA GPU (CUDA Fallback Mode)")
        else:
            device = torch.device("cpu")
            logger.info("Running on CPU (Hardware Simulation Mode)")
        return device


def calculate_mfu(model_params: int, batch_size: int, seq_len: int, step_time_sec: float, peak_tflops: float = 190.0) -> float:
    """
    Computes Model FLOPs Utilization (MFU) on Trainium2.
    Trn2 single chip theoretical BF16 peak is ~190 TFLOPs.
    Standard Transformer FLOPs per token = 6 * N_params (forward + backward).
    """
    flops_per_step = 6.0 * model_params * batch_size * seq_len
    achieved_tflops = (flops_per_step / step_time_sec) / 1e12
    mfu = (achieved_tflops / peak_tflops) * 100.0
    return max(0.0, min(100.0, mfu))


def run_30min_speedrun(args):
    device = get_device()
    is_xla = (device.type == "xla")
    
    # 1. Initialize Model Configuration
    if args.model_type == "small":
        config = get_speedrun_small_config()
    elif args.model_type == "moe":
        config = get_speedrun_moe_config()
    else:
        config = get_speedrun_base_config()
        
    config.max_seq_len = args.seq_len
    logger.info(f"Initialized Config: {args.model_type.upper()} | Dim: {config.dim} | Layers: {config.n_layers} | Heads: {config.n_heads} | SeqLen: {config.max_seq_len}")
    
    # 2. Instantiate Model
    if config.is_moe:
        model = Trn2MoELM(config).to(device)
    else:
        model = Trn2TransformerLM(config).to(device)
        
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {num_params / 1e6:.2f}M ({num_params:,} parameters)")
    
    # 3. Setup Optimizers
    muon_opt, adamw_opt = create_frontier_optimizer(
        model,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.weight_decay
    )
    
    # 4. Data Loaders
    train_loader, val_loader = create_speedrun_dataloaders(
        vocab_size=config.vocab_size,
        seq_len=config.max_seq_len,
        batch_size=args.batch_size,
        val_samples=32
    )
    train_iter = iter(train_loader)
    
    # 5. Speedrun Timer & State
    max_duration_sec = args.duration_sec
    logger.info(f"=== Starting 30-Minute Speedrun (Budget: {max_duration_sec:.1f}s = {max_duration_sec/60:.1f} min) ===")
    
    start_time = time.time()
    step = 0
    total_tokens = 0
    best_val_bpb = float("inf")
    
    # Checkpoints directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.checkpoint_dir, f"frontier_{args.model_type}_best.pt")
    
    # Estimated total steps for LR scheduling
    est_total_steps = int(max_duration_sec / 0.15)  # approximate 150ms per step
    warmup_steps = int(est_total_steps * 0.03)
    decay_steps = int(est_total_steps * 0.20)
    
    model.train()
    
    try:
        while True:
            elapsed_sec = time.time() - start_time
            if elapsed_sec >= max_duration_sec:
                logger.info(f"Time budget of {max_duration_sec}s reached. Stopping training.")
                break
                
            step_start = time.time()
            inputs, targets, _ = next(train_iter)
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Update Learning Rates (WSD Schedule)
            curr_muon_lr = get_lr_wsd(step, est_total_steps, warmup_steps, decay_steps, args.muon_lr, min_lr=args.muon_lr * 0.05)
            curr_adamw_lr = get_lr_wsd(step, est_total_steps, warmup_steps, decay_steps, args.adamw_lr, min_lr=args.adamw_lr * 0.05)
            
            for g in muon_opt.param_groups:
                g["lr"] = curr_muon_lr
            for g in adamw_opt.param_groups:
                g["lr"] = curr_adamw_lr
                
            # Zero Gradients
            muon_opt.zero_grad(set_to_none=True)
            adamw_opt.zero_grad(set_to_none=True)
            
            # Forward & Backward Pass
            logits, loss = model(inputs, targets=targets)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer Step
            muon_opt.step()
            adamw_opt.step()
            
            if is_xla:
                import torch_xla.core.xla_model as xm
                xm.mark_step()
                
            step_time = time.time() - step_start
            step_tokens = inputs.shape[0] * inputs.shape[1]
            total_tokens += step_tokens
            tokens_per_sec = step_tokens / max(1e-5, step_time)
            mfu = calculate_mfu(num_params, inputs.shape[0], inputs.shape[1], step_time)
            
            # Periodic Telemetry Log
            if step % args.log_interval == 0:
                mins_left = max(0.0, (max_duration_sec - (time.time() - start_time)) / 60.0)
                logger.info(
                    f"Step {step:05d} | Loss: {loss.item():.4f} | Tok/s: {tokens_per_sec:,.0f} | "
                    f"MFU: {mfu:.1f}% | Muon LR: {curr_muon_lr:.5f} | Time Left: {mins_left:.2f}m"
                )
                
            # Periodic Validation Evaluation
            if step > 0 and step % args.eval_interval == 0:
                val_bpb, val_loss, val_tokens = evaluate_bpb(model, val_loader, device=device, max_batches=10)
                logger.info(f"--> [EVAL Step {step}] Val Loss: {val_loss:.4f} | Val BPB: {val_bpb:.4f} bits/byte")
                
                if val_bpb < best_val_bpb:
                    best_val_bpb = val_bpb
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "val_bpb": val_bpb,
                        "val_loss": val_loss,
                        "elapsed_sec": time.time() - start_time,
                        "total_tokens": total_tokens
                    }, best_ckpt_path)
                    logger.info(f"*** New Leaderboard Best Val BPB: {best_val_bpb:.4f}! Saved checkpoint to {best_ckpt_path} ***")
                    
                model.train()
                
            step += 1
            
    except KeyboardInterrupt:
        logger.info("Speedrun interrupted by user. Finalizing evaluation...")

    total_training_time = time.time() - start_time
    logger.info(f"\n==========================================")
    logger.info(f"SPEEDRUN FINISHED IN {total_training_time:.2f} SECONDS")
    logger.info(f"Total Steps Completed: {step:,}")
    logger.info(f"Total Tokens Processed: {total_tokens:,}")
    logger.info(f"Average Throughput: {total_tokens / total_training_time:,.0f} tokens/sec")
    
    # Final Validation BPB Evaluation
    final_val_bpb, final_val_loss, val_toks = evaluate_bpb(model, val_loader, device=device, max_batches=20)
    logger.info(f"Final Validation Loss: {final_val_loss:.4f}")
    logger.info(f"Final Validation BPB:  {final_val_bpb:.4f} bits/byte")
    logger.info(f"Best Observed Val BPB: {min(best_val_bpb, final_val_bpb):.4f} bits/byte")
    logger.info(f"==========================================\n")
    
    return {
        "final_val_bpb": final_val_bpb,
        "best_val_bpb": min(best_val_bpb, final_val_bpb),
        "total_tokens": total_tokens,
        "total_steps": step,
        "training_time_sec": total_training_time
    }


def main():
    parser = argparse.ArgumentParser(description="AWS Trainium Frontier 30-Minute Speedrun")
    parser.add_argument("--model-type", type=str, default="base", choices=["small", "base", "moe"], help="Model architecture")
    parser.add_argument("--duration-sec", type=float, default=1800.0, help="Wall-clock budget in seconds (default: 1800s = 30m)")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size per NeuronCore")
    parser.add_argument("--seq-len", type=int, default=2048, help="Context length")
    parser.add_argument("--muon-lr", type=float, default=0.02, help="Peak Muon learning rate")
    parser.add_argument("--adamw-lr", type=float, default=0.001, help="Peak AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between log messages")
    parser.add_argument("--eval-interval", type=int, default=50, help="Steps between validation evaluations")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick 10-second verification run")
    
    args = parser.parse_args()
    if args.dry_run:
        args.duration_sec = 10.0
        args.log_interval = 2
        args.eval_interval = 5
        logger.info("Executing DRY-RUN (10 seconds verification)")
        
    run_30min_speedrun(args)


if __name__ == "__main__":
    main()
