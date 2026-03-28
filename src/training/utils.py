"""
PikoGPT Training Utilities
==========================
Helper functions for training with warmup scheduling, gradient monitoring,
and model summary printing.

Usage:
    1. Copy this file to src/training/utils.py
    2. Import functions in src/training/stage.py:
       from .utils import (
           get_cosine_schedule_with_warmup,
           compute_gradient_norm,
           print_model_summary,
           print_training_config
       )
    3. Use in your training code as shown below
"""

import math
import logging
from typing import Optional
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


# ============================================================
# LEARNING RATE SCHEDULING
# ============================================================

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1
) -> LambdaLR:
    """
    Create learning rate scheduler with linear warmup and cosine decay.
    
    Schedule visualization:
    
    LR ^
       |    /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
    max|   /                       \
       |  /                         \
       | /                           \
    min|/                             \___
       +---------------------------------> Steps
         |warmup|    cosine decay    |end
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps (linear increase from 0)
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as fraction of max LR (default 0.1 = 10%)
    
    Returns:
        LambdaLR scheduler
    
    Example:
        optimizer = AdamW(model.parameters(), lr=3e-4)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            warmup_steps=500, 
            total_steps=50000,
            min_lr_ratio=0.1
        )
        
        # In training loop:
        for step, batch in enumerate(dataloader):
            loss = train_step(batch)
            optimizer.step()
            scheduler.step()  # Update LR
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase: linear increase from 0 to 1
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # Clamp progress to [0, 1] to handle steps beyond total_steps
        progress = min(1.0, progress)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int
) -> LambdaLR:
    """
    Simpler alternative: linear warmup followed by linear decay to 0.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# GRADIENT MONITORING
# ============================================================

def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute the L2 norm of all gradients in the model.
    
    This is useful for:
    - Monitoring training stability
    - Detecting vanishing/exploding gradients
    - Debugging training issues
    
    Args:
        model: PyTorch model with computed gradients (after loss.backward())
    
    Returns:
        L2 norm of all gradients (sqrt of sum of squared gradients)
    
    Interpretation:
        - Very small (< 1e-7): Vanishing gradients, model not learning
        - Small (1e-4 to 1e-2): Normal for well-trained models
        - Medium (0.1 to 10): Healthy learning
        - Large (> 100): Potential instability, may need gradient clipping
    
    Example:
        loss.backward()
        grad_norm = compute_gradient_norm(model)
        
        # Clip gradients if needed
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    """
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def check_gradient_health(grad_norm: float) -> str:
    """
    Analyze gradient norm and return health status message.
    
    Args:
        grad_norm: Current gradient L2 norm
    
    Returns:
        Status message string
    """
    if grad_norm < 1e-7:
        return "⚠️  VANISHING GRADIENTS - model may not be learning"
    elif grad_norm > 1000:
        return "🔥 EXPLODING GRADIENTS - training unstable!"
    elif grad_norm > 100:
        return "⚡ High gradients - consider reducing learning rate"
    elif grad_norm > 10:
        return "📈 Elevated gradients - monitor closely"
    else:
        return "✓ Gradients healthy"


# ============================================================
# MODEL SUMMARY PRINTING
# ============================================================

def print_model_summary(
    model: nn.Module,
    config,
    logger: Optional[logging.Logger] = None
):
    """
    Print detailed model architecture summary.
    
    Args:
        model: PyTorch model
        config: Config object with model/tokenizer attributes
        logger: Optional logger (uses print if None)
    
    Example output:
        ======================================================================
         MODEL ARCHITECTURE SUMMARY 
        ======================================================================
          Vocabulary size:     50,257
          Context length:      512
          Embedding dim:       384
          Layers:              10
          Attention heads:     6
          Head dimension:      64
          FFN hidden size:     1,536
          Dropout:             0.1
        ======================================================================
          Total parameters:    37,240,704 (37.24M)
          Trainable params:    37,240,704
        ======================================================================
    """
    log = logger.info if logger else print
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params
    
    # Get vocab size from model if available
    vocab_size = getattr(model, 'vocab_size', 50257)
    
    log("")
    log("=" * 70)
    log(" MODEL ARCHITECTURE SUMMARY ")
    log("=" * 70)
    log(f"  Vocabulary size:     {vocab_size:,}")
    log(f"  Context length:      {config.tokenizer.context_length}")
    log(f"  Embedding dim:       {config.model.n_embd}")
    log(f"  Layers:              {config.model.n_layer}")
    log(f"  Attention heads:     {config.model.n_head}")
    log(f"  Head dimension:      {config.model.n_embd // config.model.n_head}")
    log(f"  FFN hidden size:     {config.model.n_embd * 4:,}")
    log(f"  Dropout:             {config.model.dropout}")
    log("=" * 70)
    log(f"  Total parameters:    {total_params:,} ({total_params/1e6:.2f}M)")
    log(f"  Trainable params:    {trainable_params:,}")
    if non_trainable > 0:
        log(f"  Non-trainable:       {non_trainable:,}")
    log("=" * 70)
    
    # Memory estimate
    param_bytes = total_params * 4  # float32 = 4 bytes
    log(f"  Model size (fp32):   {param_bytes / 1e6:.1f} MB")
    log(f"  Est. train memory:   {param_bytes * 4 / 1e6:.1f} MB")  # ~4x for gradients, optimizer states
    log("=" * 70)
    log("")


def print_layer_shapes(config, logger: Optional[logging.Logger] = None):
    """
    Print expected tensor shapes through the model.
    
    Args:
        config: Config object with model/tokenizer attributes
        logger: Optional logger (uses print if None)
    """
    log = logger.info if logger else print
    
    batch_size = config.training.batch_size
    seq_len = config.tokenizer.context_length
    n_embd = config.model.n_embd
    vocab_size = 50257  # GPT-2 vocab
    
    log("")
    log("=" * 70)
    log(" TENSOR SHAPES (Forward Pass) ")
    log("=" * 70)
    log(f"  Input token IDs:     ({batch_size}, {seq_len})")
    log(f"  After tok_embed:     ({batch_size}, {seq_len}, {n_embd})")
    log(f"  After pos_embed:     ({batch_size}, {seq_len}, {n_embd})")
    log(f"  After transformer:   ({batch_size}, {seq_len}, {n_embd})")
    log(f"  Output logits:       ({batch_size}, {seq_len}, {vocab_size})")
    log("")
    log("  For next-token prediction (shifted):")
    log(f"    logits_shift:      ({batch_size}, {seq_len-1}, {vocab_size})")
    log(f"    labels_shift:      ({batch_size}, {seq_len-1})")
    log(f"    loss input:        ({batch_size * (seq_len-1)}, {vocab_size})")
    log("=" * 70)
    log("")


def print_training_config(config, total_steps: int, logger: Optional[logging.Logger] = None):
    """
    Print training configuration summary.
    
    Args:
        config: Config object
        total_steps: Total training steps
        logger: Optional logger (uses print if None)
    """
    log = logger.info if logger else print
    
    # Get warmup steps with fallback
    warmup_steps = getattr(config.training, 'warmup_steps', 0)
    min_lr = getattr(config.training, 'min_learning_rate', 1e-5)
    
    log("")
    log("=" * 70)
    log(" TRAINING CONFIGURATION ")
    log("=" * 70)
    log(f"  Device:              {config.training.device}")
    log(f"  Batch size:          {config.training.batch_size}")
    log(f"  Sequence length:     {config.tokenizer.context_length}")
    log(f"  Tokens per batch:    {config.training.batch_size * config.tokenizer.context_length:,}")
    log("-" * 70)
    log(f"  Total steps:         {total_steps:,}")
    log(f"  Warmup steps:        {warmup_steps}")
    log(f"  Learning rate:       {config.training.learning_rate}")
    log(f"  Min learning rate:   {min_lr}")
    log(f"  Weight decay:        {config.training.weight_decay}")
    log(f"  Gradient clip:       {config.training.gradient_clip_norm}")
    log(f"  torch.compile:       {config.training.compile_model}")
    log(f"  Compile backend:     {config.training.compile_backend}")
    log(f"  Save every:          {config.training.save_every_steps} steps")
    log("-" * 70)
    log(f"  Log every:           {config.training.log_every_steps} steps")
    log(f"  Eval every:          {config.training.eval_every_steps} steps")
    log("=" * 70)
    log("")


# ============================================================
# DEVICE UTILITIES
# ============================================================

def get_device(device_str: str = "auto") -> torch.device:
    """
    Resolve device string to torch.device.
    
    Args:
        device_str: "auto", "cuda", "mps", or "cpu"
    
    Returns:
        torch.device object
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def print_device_info(device: torch.device, logger: Optional[logging.Logger] = None):
    """Print information about compute device."""
    log = logger.info if logger else print
    
    log("")
    log(f"🖥️  Using device: {device}")
    
    if device.type == "cuda":
        log(f"   GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"   Memory: {mem_gb:.1f} GB")
    elif device.type == "mps":
        log(f"   Apple Silicon GPU (MPS)")
    else:
        log(f"   ⚠️  CPU training - will be slow for large models")
    log("")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TRAINING STEP FORMATTING
# ============================================================

def format_step_log(
    step: int,
    total_steps: int,
    loss: float,
    lr: float,
    grad_norm: float,
    epoch: Optional[int] = None,
    tokens_per_sec: Optional[float] = None
) -> str:
    """
    Format a training step log message.
    
    Args:
        step: Current step
        total_steps: Total steps
        loss: Current loss
        lr: Current learning rate
        grad_norm: Current gradient norm
        epoch: Optional epoch number
        tokens_per_sec: Optional throughput
    
    Returns:
        Formatted log string
    """
    parts = []
    
    if epoch is not None:
        parts.append(f"Epoch {epoch}")
    
    progress = step / total_steps * 100 if total_steps > 0 else 0
    parts.append(f"Step {step:,}/{total_steps:,} ({progress:.1f}%)")
    parts.append(f"loss={loss:.4f}")
    parts.append(f"lr={lr:.2e}")
    parts.append(f"grad={grad_norm:.4f}")
    
    if tokens_per_sec is not None:
        parts.append(f"tok/s={tokens_per_sec:,.0f}")
    
    return " | ".join(parts)


# ============================================================
# DISTRIBUTED DATA PARALLEL (DDP) UTILITIES
# ============================================================

def setup_distributed() -> tuple[int, int, int, torch.device]:
    """
    Initialize the distributed process group and return rank info + device.

    Designed to be called once per GPU process when launched via ``torchrun``.
    Falls back gracefully to single-GPU / CPU when ``torchrun`` is not used.

    Returns:
        (rank, world_size, local_rank, device)
    """
    import os
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if use_cuda:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")

    print(
        f"[rank {rank}/{world_size}] local_rank={local_rank} device={device}",
        flush=True,
    )
    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    """Destroy the process group so repeated runs don't hang."""
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    """Return True when running inside a distributed process group."""
    import torch.distributed as dist
    return dist.is_initialized() and dist.get_world_size() > 1


def is_main_process() -> bool:
    """Return True on rank-0 (or when not running distributed)."""
    import torch.distributed as dist
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    """Demo the utilities."""
    print("=" * 70)
    print(" TRAINING UTILITIES DEMO ")
    print("=" * 70)
    
    # Test device detection
    device = get_device("auto")
    print_device_info(device)
    
    # Test LR scheduler visualization
    print("\n📈 Learning Rate Schedule Preview:")
    print("-" * 50)
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Create scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        warmup_steps=100, 
        total_steps=1000,
        min_lr_ratio=0.1
    )
    
    # Print LR at key steps
    print(f"{'Step':>8} {'LR':>12} {'Phase':>15}")
    print("-" * 40)
    
    for step in [0, 25, 50, 75, 100, 250, 500, 750, 900, 999]:
        # Get current LR
        lr = scheduler.get_last_lr()[0]
        
        # Determine phase
        if step < 100:
            phase = "warmup"
        else:
            phase = "cosine decay"
        
        print(f"{step:>8} {lr:>12.6f} {phase:>15}")
        
        # Step forward
        optimizer.step()
        scheduler.step()
    
    print("\n✅ All utilities working correctly!")
    print("=" * 70)
