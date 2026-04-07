"""
PikoGPT Hyperparameter Tuning with Optuna
==========================================

This script uses Optuna to find optimal hyperparameters for PikoGPT.

Usage:
    # Run hyperparameter search (20 trials)
    python -m src.tuning.optuna_search --n-trials 20
    
    # View results in dashboard
    optuna-dashboard sqlite:///optuna_pikogpt.db --host 127.0.0.1 --port 8080

What we tune:
    - learning_rate: Peak learning rate
    - batch_size: Training batch size
    - n_layer: Number of transformer layers
    - n_head: Number of attention heads
    - n_embd: Embedding dimension
    - dropout: Dropout probability
    - warmup_steps: LR warmup steps
    - weight_decay: AdamW weight decay

What we keep fixed:
    - vocab_size: 50,257 (GPT-2 tokenizer)
    - context_length: 512 (or smaller for faster tuning)
    - max_train_steps: Short for tuning (e.g., 500 steps)
"""

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# CONFIGURATION
# ============================================================

# Fixed parameters (not tuned)
VOCAB_SIZE = 50257  # GPT-2 tokenizer
CONTEXT_LENGTH = 256  # Shorter for faster tuning (use 512 for final)
MAX_PARAM_BUDGET = 40_000_000  # 40M parameter limit

# Tuning parameters
TUNE_TRAIN_STEPS = 500  # Short training for each trial
TUNE_EVAL_STEPS = 50  # Evaluation steps
TUNE_EVAL_INTERVAL = 100  # Evaluate every N steps
TUNE_LOG_INTERVAL = 50  # Log every N steps

# Database for Optuna
DB_PATH = Path("optuna_pikogpt.db")
STORAGE = f"sqlite:///{DB_PATH}"
STUDY_NAME = "pikogpt_hyperparameter_search"


# ============================================================
# MODEL DEFINITION (Simplified for tuning)
# ============================================================

class TuningTransformerLM(nn.Module):
    """
    Simplified transformer for hyperparameter tuning.
    Uses PyTorch's built-in TransformerEncoder for speed.
    """
    
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        context_length: int,
        dropout: float,
    ):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        
        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(context_length, n_embd)
        self.drop = nn.Dropout(dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        # Output
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.tok_emb.weight
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        
        # Embeddings
        tok_emb = self.tok_emb(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        
        # Transformer
        x = self.transformer(x, mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1),
                targets.view(B * T),
                ignore_index=-100,
            )
        
        return logits, loss


def count_parameters(n_embd: int, n_layer: int, vocab_size: int = VOCAB_SIZE, context_length: int = CONTEXT_LENGTH) -> int:
    """Estimate parameter count for a given configuration."""
    # Token embeddings (tied with output, so count once)
    tok_emb = vocab_size * n_embd
    
    # Position embeddings
    pos_emb = context_length * n_embd
    
    # Per transformer layer:
    # - Attention: 4 * n_embd^2 (Q, K, V, O projections)
    # - FFN: 2 * n_embd * (4 * n_embd) = 8 * n_embd^2
    # - LayerNorms: 4 * n_embd
    per_layer = 4 * n_embd * n_embd + 8 * n_embd * n_embd + 4 * n_embd
    all_layers = n_layer * per_layer
    
    # Final LayerNorm
    final_ln = 2 * n_embd
    
    return tok_emb + pos_emb + all_layers + final_ln


# ============================================================
# DATA LOADING (Simplified)
# ============================================================

def load_tuning_data(data_path: str = "data/processed/openwebtext_clean"):
    """
    Load a small subset of data for hyperparameter tuning.
    """
    from datasets import load_from_disk
    from transformers import GPT2TokenizerFast
    
    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_from_disk(data_path)
    
    # Take a small subset for tuning
    if len(dataset) > 5000:
        dataset = dataset.select(range(5000))
    
    return dataset, tokenizer


def create_tuning_dataloader(
    dataset,
    tokenizer,
    batch_size: int,
    context_length: int,
    split: str = "train",
):
    """
    Create a DataLoader for tuning.
    """
    from torch.utils.data import Dataset as TorchDataset
    
    class TokenizedDataset(TorchDataset):
        def __init__(self, texts, tokenizer, context_length):
            self.examples = []
            
            for text in texts:
                tokens = tokenizer.encode(text, truncation=True, max_length=context_length + 1)
                if len(tokens) >= context_length + 1:
                    self.examples.append(tokens[:context_length + 1])
        
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            tokens = self.examples[idx]
            x = torch.tensor(tokens[:-1], dtype=torch.long)
            y = torch.tensor(tokens[1:], dtype=torch.long)
            return {"input_ids": x, "labels": y}
    
    # Split dataset
    n = len(dataset)
    split_idx = int(0.9 * n)
    
    if split == "train":
        texts = [dataset[i]["text"] for i in range(split_idx)]
    else:
        texts = [dataset[i]["text"] for i in range(split_idx, n)]
    
    torch_dataset = TokenizedDataset(texts, tokenizer, context_length)
    
    return DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=True,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: DataLoader, max_steps: int = TUNE_EVAL_STEPS) -> float:
    """Evaluate model and return average loss."""
    model.eval()
    losses = []
    
    for step, batch in enumerate(dataloader):
        if step >= max_steps:
            break
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        _, loss = model(input_ids, labels)
        losses.append(loss.item())
    
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


# ============================================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================================

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function.
    
    Samples hyperparameters, trains the model briefly, and returns validation loss.
    """
    
    # ========== 1. SAMPLE HYPERPARAMETERS ==========
    
    # Model architecture
    n_embd = trial.suggest_categorical("n_embd", [256, 320, 384, 448])
    n_layer = trial.suggest_int("n_layer", 4, 12)
    n_head = trial.suggest_categorical("n_head", [4, 6, 8])
    dropout = trial.suggest_float("dropout", 0.0, 0.2)
    
    # Training hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.001, 0.1, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.01, 0.1)
    
    # ========== 2. VALIDATE CONFIGURATION ==========
    
    # Check head dimension is valid
    if n_embd % n_head != 0:
        raise optuna.TrialPruned("n_embd must be divisible by n_head")
    
    # Check parameter budget
    estimated_params = count_parameters(n_embd, n_layer)
    if estimated_params > MAX_PARAM_BUDGET:
        raise optuna.TrialPruned(f"Exceeds {MAX_PARAM_BUDGET/1e6:.0f}M parameter budget")
    
    trial.set_user_attr("estimated_params", estimated_params)
    trial.set_user_attr("estimated_params_M", estimated_params / 1e6)
    
    # ========== 3. CREATE MODEL ==========
    
    model = TuningTransformerLM(
        vocab_size=VOCAB_SIZE,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        context_length=CONTEXT_LENGTH,
        dropout=dropout,
    ).to(device)
    
    print(f"\nTrial {trial.number}: {model.n_params/1e6:.2f}M params | "
          f"n_embd={n_embd}, n_layer={n_layer}, n_head={n_head}, "
          f"lr={learning_rate:.2e}, bs={batch_size}")
    
    # ========== 4. LOAD DATA ==========
    
    try:
        dataset, tokenizer = load_tuning_data()
        train_loader = create_tuning_dataloader(
            dataset, tokenizer, batch_size, CONTEXT_LENGTH, split="train"
        )
        val_loader = create_tuning_dataloader(
            dataset, tokenizer, batch_size, CONTEXT_LENGTH, split="val"
        )
    except Exception as e:
        print(f"Data loading failed: {e}")
        raise optuna.TrialPruned(f"Data loading failed: {e}")
    
    # ========== 5. SETUP TRAINING ==========
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Simple linear warmup + constant LR
    warmup_steps = int(TUNE_TRAIN_STEPS * warmup_ratio)
    
    def get_lr(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    
    # ========== 6. TRAINING LOOP ==========
    
    model.train()
    global_step = 0
    train_iter = iter(train_loader)
    
    for step in range(TUNE_TRAIN_STEPS):
        # Get batch (cycle through data if needed)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        _, loss = model(input_ids, labels)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        global_step += 1
        
        # ========== 7. INTERMEDIATE EVALUATION (for pruning) ==========
        
        if global_step % TUNE_EVAL_INTERVAL == 0:
            val_loss = evaluate_model(model, val_loader)
            
            # Report to Optuna for pruning
            trial.report(val_loss, global_step)
            
            # Check if trial should be pruned
            if trial.should_prune():
                print(f"  Trial {trial.number} pruned at step {global_step} (val_loss={val_loss:.4f})")
                raise optuna.TrialPruned()
            
            if global_step % TUNE_LOG_INTERVAL == 0:
                print(f"  Step {global_step}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")
    
    # ========== 8. FINAL EVALUATION ==========
    
    final_val_loss = evaluate_model(model, val_loader, max_steps=100)
    print(f"  Trial {trial.number} complete: final_val_loss={final_val_loss:.4f}")
    
    return final_val_loss


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PikoGPT Hyperparameter Tuning")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--study-name", type=str, default=STUDY_NAME, help="Optuna study name")
    parser.add_argument("--resume", action="store_true", help="Resume from existing study")
    args = parser.parse_args()
    
    print("=" * 70)
    print(" PikoGPT HYPERPARAMETER TUNING ")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Trials: {args.n_trials}")
    print(f"Train steps per trial: {TUNE_TRAIN_STEPS}")
    print(f"Context length: {CONTEXT_LENGTH}")
    print(f"Parameter budget: {MAX_PARAM_BUDGET/1e6:.0f}M")
    print(f"Database: {DB_PATH.resolve()}")
    print("=" * 70)
    
    # Create or load study
    study = optuna.create_study(
        direction="minimize",  # Minimize validation loss
        study_name=args.study_name,
        storage=STORAGE,
        load_if_exists=args.resume,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,  # Don't prune first 5 trials
            n_warmup_steps=100,  # Don't prune before 100 steps
        ),
    )
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    # Print results
    print("\n" + "=" * 70)
    print(" OPTIMIZATION RESULTS ")
    print("=" * 70)
    
    print(f"\nBest validation loss: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    if "estimated_params_M" in study.best_trial.user_attrs:
        print(f"\nEstimated parameters: {study.best_trial.user_attrs['estimated_params_M']:.2f}M")
    
    print(f"\nDatabase saved at: {DB_PATH.resolve()}")
    print(f"\nTo view dashboard, run:")
    print(f"  optuna-dashboard sqlite:///{DB_PATH.resolve()} --host 127.0.0.1 --port 8080")
    print(f"\nThen open: http://127.0.0.1:8080")


if __name__ == "__main__":
    main()