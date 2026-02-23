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
├── notebooks/
│   └── 01_EDA.ipynb              # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       └── preprocessing.py       # Data preprocessing pipeline
├── data/                          # Not tracked in git (see .gitignore)
│   ├── raw/                       # Downloaded test data
│   └── processed/                 # Cleaned training data
├── main.py                        # Main entry point
├── pyproject.toml                 # Project dependencies
└── README.md
```

## Stages

| Stage | Command | Status | Description |
|-------|---------|--------|-------------|
| `preprocess` | `python main.py --stage preprocess` | ✅ Implemented | Clean and filter OpenWebText |
| `train` | `python main.py --stage train` | 🔲 TODO | Pretrain the language model |
| `inference` | `python main.py --stage inference` | 🔲 TODO | Generate text from trained model |

## Usage

### Preprocessing
```bash
# Basic usage (100K samples)
python main.py --stage preprocess --num-samples 100000

# Custom configuration
python main.py --stage preprocess \
    --num-samples 50000 \
    --seed 123 \
    --test-data-path "data/raw/NLP26_OWT_eval/test" \
    --output-path "data/processed/my_dataset"
```

### Training (Coming Soon)
```bash
python main.py --stage train --config configs/default.yaml
```

### Inference (Coming Soon)
```bash
python main.py --stage inference \
    --checkpoint checkpoints/model.pt \
    --prompt "Once upon a time" \
    --max-tokens 100 \
    --temperature 0.8
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

## Team

**Startup Name:** FunkyAI

| Member   | Role |
|--------  |------|
| Filipp   | TBD |
| Roman    | TBD |
| Arabella | TBD |


## License

This project is for educational purposes as part of the University of St. Gallen NLP course.