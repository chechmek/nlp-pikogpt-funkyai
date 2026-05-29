# PikoGPT — FunkyAI

A small decoder-only language model built from scratch for the NLP with LLMs course (Spring 2026).

PikoGPT is a GPT-2 style transformer trained on 2 million OpenWebText documents. The repo implements the complete LLM pipeline: data preprocessing, pretraining, supervised fine-tuning (SFT), direct preference optimization (DPO), evaluation, and inference / chat.

## Key Results

| Metric | Value |
|--------|-------|
| Parameters | 37.4M |
| Eval Perplexity (5 epochs) | 38.33 |
| WikiText-103 Perplexity | 103.99 |
| OWT Test Perplexity | 79.64 |
| Training Time | ~3 hours (8×V100) |

## Technical Constraints

As per the PikoGPT Challenge rules:

- **Architecture:** Decoder-only (no MoE)
- **Model size:** Max 40M parameters
- **Context length:** Max 1024 tokens
- **Tokenizer:** GPT-2 tokenizer (fixed)
- **Training data:** OpenWebText (provided subset)
- **Compute budget:** 2 × 24h on 8×V100

## Model Architecture

The primary model (`configs/train_large.toml`) is a GPT-2 style decoder-only transformer:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_embd` | 384 | Embedding dimension |
| `n_layer` | 10 | Transformer blocks |
| `n_head` | 6 | Attention heads |
| `head_dim` | 64 | Per-head dimension |
| `context_length` | 1024 | Max sequence length |
| `vocab_size` | 50,257 | GPT-2 tokenizer |
| **Total params** | **~37M** | Within 40M budget |

### Architecture Diagram

```
Input Token IDs
       ↓
┌──────────────────┐
│ Token Embedding  │ (50,257 × 384)
│ + Position Embed │ (1024 × 384)
├──────────────────┤
│ Dropout (0.1)    │
├──────────────────┤
│                  │
│ Transformer Block│ ×10
│ ├─ LayerNorm     │
│ ├─ Multi-Head    │
│ │  Attention (6) │
│ ├─ Residual +    │
│ ├─ LayerNorm     │
│ ├─ FFN (384→1536→384)
│ └─ Residual +    │
│                  │
├──────────────────┤
│ Final LayerNorm  │
├──────────────────┤
│ Output Projection│ (384 → 50,257)
│ (tied weights)   │
└──────────────────┘
       ↓
     Logits
```

### Training Features

- **LR schedule:** Linear warmup + cosine decay
- **Optimizer:** AdamW with weight decay
- **Gradient clipping:** Max norm 1.0
- **Logging:** JSONL metrics + console output
- **Checkpointing:** Self-contained checkpoints (architecture + weights)

## Benchmark Results

### Pretraining (5 epochs)

| Metric | Value |
|--------|-------|
| Eval Perplexity | 38.33 |
| WikiText-103 Perplexity | 103.99 |
| OWT Test Perplexity | 79.64 |

### SFT Model Benchmarks

| Benchmark | Accuracy | Random Baseline |
|-----------|----------|-----------------|
| HellaSwag | 24% | 25% |
| WinoGrande | 52% | 50% |
| OpenBookQA | 26% | 25% |
| LAMBADA | 0% | 0% |

Note: results are not statistically significant from the random baseline (p > 0.05, n = 50).

## EDA Findings

Based on analysis of 10,000 OpenWebText samples:

| Issue | Prevalence | Action |
|-------|------------|--------|
| Non-English content | 0.3% | Filter using langdetect |
| HTML tags | 1.3% | Remove with regex |
| URLs | 7.2% | Remove with regex |
| Code snippets | 1.1% | Remove code blocks |
| Quality issues | 1.0% | Filter corrupted documents |

**Preprocessing keep rate: ~97.8%**

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

## Quick Start

### 1. Clone and Setup
```bash
git clone <repo-url>
cd nlp-pikogpt-funkyai

uv venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate   # macOS / Linux

uv pip install -e .
```

### 2. Download Test Data (Required)

Download the NLP26 test split to prevent data leakage during training:

1. Go to: https://drive.switch.ch/index.php/s/6TLGQFEIkAPJ72K
2. Download the `NLP26_OWT_eval/test` folder
3. Save to: `data/raw/NLP26_OWT_eval/test/`

Your folder structure should look like:
```
data/
└── raw/
    └── NLP26_OWT_eval/
        └── test/
            ├── data-00000-of-00004.arrow
            ├── data-00001-of-00004.arrow
            ├── data-00002-of-00004.arrow
            ├── data-00003-of-00004.arrow
            ├── dataset_info.json
            └── state.json
```

### 3. Run Preprocessing

```bash
python main.py --stage preprocess --num-samples 2000000 --output-path "data/processed/openwebtext_clean"
```

## Project Structure
```
nlp-pikogpt-funkyai/
├── configs/
│   ├── train_smoke.toml           # Tiny config for CI / smoke tests
│   ├── train_default.toml         # Small model for CPU testing (~16M params)
│   ├── train_medium_test.toml     # Medium model for iteration (~28M params)
│   ├── train_large.toml           # Primary config (~37M params) ⭐
│   ├── train_deep.toml            # Deep variant (~34M params)
│   ├── train_fullcontext.toml     # 1024-context variant (~33M params)
│   └── sft_default.toml           # SFT defaults
├── src/
│   ├── data/
│   │   └── preprocessing.py       # Data preprocessing pipeline
│   ├── training/
│   │   ├── config.py              # Pydantic config models
│   │   ├── stage.py               # Pretraining loop
│   │   └── utils.py               # LR scheduling, gradient monitoring
│   ├── sft/
│   │   └── stage.py               # Supervised fine-tuning
│   ├── dpo/
│   │   └── stage.py               # Direct preference optimization
│   ├── inference/
│   │   └── stage.py               # Text generation pipeline
│   ├── chat/
│   │   └── stage.py               # Gradio chat UI
│   ├── evaluation/
│   │   └── stage.py               # Standalone benchmark evaluation
│   └── tuning/
│       └── optuna_search.py       # Hyperparameter search
├── scripts/
│   └── full_training_run.sh       # End-to-end training pipeline
├── runs/                          # Training outputs (not tracked)
├── data/                          # Datasets (not tracked)
├── main.py                        # CLI entry point
└── pyproject.toml                 # Project dependencies
```

## Stages

All pipeline stages are driven through `python main.py --stage <name>`:

| Stage | Description |
|-------|-------------|
| `preprocess` | Clean and filter OpenWebText |
| `train` | Pretrain the language model |
| `sft` | Supervised instruction tuning from a base checkpoint |
| `dpo` | Preference optimization from an SFT checkpoint |
| `inference` | Generate text from a trained checkpoint |
| `evaluate` | Run standalone benchmark evaluation (OWT + WikiText) |
| `chat` | Local Gradio chat UI for a checkpoint |

## Usage

### Preprocessing
```bash
# Basic usage (100K samples)
python main.py --stage preprocess \
    --num-samples 100000 \
    --source-dataset-path "data/raw/openwebtext_local"

# Custom configuration
python main.py --stage preprocess \
    --num-samples 50000 \
    --seed 123 \
    --source-dataset-path "data/raw/openwebtext_local" \
    --test-data-path "data/raw/NLP26_OWT_eval/test" \
    --output-path "data/processed/my_dataset"
```

### Pretraining
```bash
# Quick test (small model, ~2 min on CPU)
python main.py --stage train --config configs/train_default.toml

# Primary model (use on GPU)
python main.py --stage train --config configs/train_large.toml

# 8-GPU single-node DDP
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    main.py --stage train --config configs/train_large.toml

# Full local pipeline on 8 GPUs (preprocess → distributed train → inference)
SOURCE_DATASET_PATH="data/raw/openwebtext_local" \
NPROC_PER_NODE=8 \
CONFIG=configs/train_large.toml \
scripts/full_training_run.sh

# Resume from a periodic checkpoint
python main.py --stage train \
    --config configs/train_large.toml \
    --resume-from runs/<run_name>/artifacts/checkpoints/step_005000.pt

# End-to-end smoke pipeline (no GPU needed)
MODE=smoke scripts/full_training_run.sh
```

### Supervised Fine-Tuning (SFT)
```bash
python main.py --stage sft \
    --base-checkpoint runs/<run_name>/artifacts/model_final.pt \
    --sft-max-samples 5000 \
    --sft-epochs 3 \
    --sft-lr 1e-4

# Use a local Alpaca-style JSONL dataset
python main.py --stage sft \
    --base-checkpoint runs/<run_name>/artifacts/model_final.pt \
    --sft-data-path data/sft/mixed_alpaca.jsonl
```

### Direct Preference Optimization (DPO)
```bash
python main.py --stage dpo \
    --base-checkpoint runs_sft/<sft_run>/artifacts/model_final_sft.pt \
    --dpo-data-path data/dpo/ultrafeedback_5k.jsonl \
    --dpo-beta 0.1 \
    --dpo-epochs 1
```

### Inference
```bash
# Interactive sampling
python main.py --stage inference \
    --checkpoint runs/<run_name>/artifacts/model_final.pt \
    --prompt "The meaning of life is" \
    --max-tokens 50 \
    --temperature 0.7

# Leaderboard mode (deterministic, outputs only generated text)
python main.py --stage inference \
    --checkpoint runs/<run_name>/artifacts/model_final.pt \
    --prompt "Question: What is the capital of France? Answer:" \
    --max-tokens 1 \
    --temperature 0 \
    --device auto \
    --leaderboard \
    --seed 0
```

### Evaluation
```bash
# Score a checkpoint on OWT test + WikiText-103
python main.py --stage evaluate \
    --checkpoint runs/<run_name>/artifacts/model_final.pt \
    --owt-test-path data/raw/NLP26_OWT_eval/test \
    --device auto
```

### Chat
```bash
# Launch the local Gradio chat UI for a checkpoint
python main.py --stage chat \
    --checkpoint runs/<run_name>/artifacts/model_final.pt \
    --server-port 7860
```
