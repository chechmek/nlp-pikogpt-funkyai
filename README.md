# PikoGPT - FunkyAI

Building a small Language Model from scratch for the NLP with LLMs course (Spring 2026).

## Project Overview

This project implements a decoder-only transformer language model (10-40M parameters) trained on OpenWebText data. The goal is to go from zero to a fully trained LLM with a chat interface in 12 weeks.

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

## Quick Start

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repo-url>
cd nlp-pikogpt-funkyai

# Create virtual environment and install dependencies
uv venv
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # Mac/Linux

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
python main.py --stage preprocess --num-samples 100000 --output-path "data/processed/openwebtext_clean"
```

This will:
- Load and hash 400K+ test sentences (prevents data leakage)
- Stream OpenWebText from HuggingFace
- Filter non-English content
- Remove HTML, URLs, code blocks, and special characters
- Save 100K clean documents for training

## Project Structure
```
nlp-pikogpt-funkyai/
├── configs/
│   ├── train_default.toml        # Small model for testing (16M params)
│   ├── train_large.toml          # Primary config (37M params) ⭐
│   ├── train_deep.toml           # Deeper model variant (34M params)
│   └── train_fullcontext.toml    # Max context length (33M params)
├── notebooks/
│   └── 01_EDA.ipynb              # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py      # Data preprocessing pipeline
│   ├── training/
│   │   ├── __init__.py
│   │   ├── config.py             # Pydantic config models
│   │   ├── stage.py              # Training loop
│   │   └── utils.py              # LR scheduling, gradient monitoring
│   └── inference/
│       ├── __init__.py
│       └── stage.py              # Text generation pipeline
├── runs/                         # Training run outputs (not tracked)
├── data/                         # Datasets (not tracked)
├── main.py                       # CLI entry point
├── pyproject.toml                # Project dependencies
├── CONTRIBUTING.md               # Team workflow & responsibilities
└── README.md
```

## Stages

| Stage | Command | Status | Description |
|-------|---------|--------|-------------|
| `preprocess` | `python main.py --stage preprocess` | ✅ Implemented | Clean and filter OpenWebText |
| `train` | `python main.py --stage train` | ✅ Implemented | Pretrain the language model |
| `inference` | `python main.py --stage inference` | ✅ Implemented | Generate text from a trained checkpoint |
| `evaluate` | `python main.py --stage evaluate` | 🔲 Planned | Run benchmarks |
| `chat` | `python main.py --stage chat` | 🔲 Planned | Interactive chat interface |

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

### Training
```bash
# Quick test (small model, ~2 min on CPU)
python main.py --stage train --config configs/train_default.toml
 
# Large model (use on GPU)
python main.py --stage train --config configs/train_large.toml

# 8-GPU single-node DDP
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    main.py --stage train --config configs/train_large.toml

# Full local pipeline on 8 GPUs (preprocess -> distributed train -> inference)
SOURCE_DATASET_PATH="data/raw/openwebtext_local" \
NPROC_PER_NODE=8 \
CONFIG=configs/train_large.toml \
scripts/full_training_run.sh

# Resume from a periodic checkpoint
python main.py --stage train \
    --config configs/train_large.toml \
    --resume-from runs/<run_name>/artifacts/checkpoints/step_005000.pt

# End-to-end smoke pipeline
MODE=smoke scripts/full_training_run.sh
```
 
### Inference
```bash
# Interactive mode
python main.py --stage inference \
    --checkpoint runs/<run_name>/artifacts/model_final.pt \
    --prompt "The meaning of life is" \
    --max-tokens 50 \
    --temperature 0.7
 
# Leaderboard mode (outputs only generated text)
python main.py --stage inference \
    --checkpoint runs/<run_name>/artifacts/model_final.pt \
    --prompt "Question: What is the capital of France? Answer:" \
    --max-tokens 1 \
    --temperature 0 \
    --device auto \
    --leaderboard \
    --seed 0
```

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

## Technical Constraints

As per the PikoGPT Challenge rules:

- **Architecture:** Decoder-only (no MoE)
- **Model size:** Max 40M parameters
- **Context length:** 1024 tokens
- **Tokenizer:** GPT-2 tokenizer
- **Training data:** OpenWebText (provided subset)
- **Compute budget:** 2x 24h on 8xV100

## Model Architecture
 
Our primary model (`train_large.toml`) uses a GPT-2 style decoder-only transformer:
 
| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_embd` | 384 | Embedding dimension |
| `n_layer` | 10 | Transformer blocks |
| `n_head` | 6 | Attention heads |
| `head_dim` | 64 | Per-head dimension |
| `context_length` | 512 | Max sequence length |
| `vocab_size` | 50,257 | GPT-2 tokenizer |
| **Total params** | **~37M** | Within 40M budget |
 
### Architecture Diagram
```
Input Token IDs
       ↓
┌──────────────────┐
│ Token Embedding  │ (50,257 → 384)
├──────────────────┤
│ Position Embed   │ (512 → 384)
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
│ ├─ FFN (384→1536)│
│ └─ Residual +    │
│                  │
├──────────────────┤
│ Final LayerNorm  │
├──────────────────┤
│ Output Projection│ (384 → 50,257)
└──────────────────┘
       ↓
Logits (next token probabilities)
```
 
## Training Features
 
- **LR Schedule:** Linear warmup + cosine decay
- **Optimizer:** AdamW with weight decay
- **Gradient Clipping:** Max norm 1.0
- **Logging:** JSONL metrics + console output
- **Checkpointing:** Self-contained checkpoints (architecture + weights)

## Project Timeline
 
| Week | Phase | Status |
|------|-------|--------|
| 1-2 | Team formation, EDA | ✅ Complete |
| 3-4 | Preprocessing, Training code | ✅ Complete |
| 5 | Architecture configs, TA check-in | ✅ Complete |
| 6-7 | GPU training, Benchmarks | 🔄 In Progress |
| 8-9 | Post-training, Evaluation | 🔲 Planned |
| 10-11 | Chat interface, Poster | 🔲 Planned |
| 12 | Final submission | 🔲 Planned |
 
## Team
 
**Startup Name:** FunkyAI
 
| Member   | Role              | Focus Area |
|----------|-------------------|------------|
| Filipp   | Data Engineer     | Preprocessing, benchmarks, evaluation |
| Roman    | Platform Engineer | Training code, inference, CLI |
| Arabella | ML Engineer       | Architecture, configs, hyperparameters |
 
See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed responsibilities and workflow.
