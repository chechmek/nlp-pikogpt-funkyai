# Train Stage

## Purpose

Train a decoder-only causal language model using:

- train/validation split
- tokenization and fixed-length sequence packing
- config-driven setup (TOML/YAML + Pydantic validation)
- structured experiment logging

Implementation: `src/training/stage.py`

## Model Backend

The stage uses a custom decoder-only model implemented with:

- `torch.nn.TransformerEncoder`
- causal self-attention mask
- tied token embedding / LM head weights
- cross-entropy next-token objective

## Run Commands

From repository root:

```bash
# Full training run
PYTHONPATH=. .venv/bin/python main.py \
  --stage train \
  --config configs/train_testrun.toml
```

```bash
# Prepare-only mode (debug pipeline, no optimizer steps)
PYTHONPATH=. .venv/bin/python main.py \
  --stage train \
  --config configs/train_testrun.toml \
  --prepare-only
```

## Config

The config file is parsed and validated by `src/training/config.py`.

Supported formats:

- `.toml`
- `.yaml` / `.yml`

Main sections:

- `experiment`: run identity and seed.
- `data`: input dataset path + split settings.
- `tokenizer`: tokenizer name and context length.
- `model`: architecture dimensions and dropout.
- `training`: optimizer/training schedule.
- `logging`: run directory and metric file names.

## Output Structure

Each run creates:

```text
runs/<experiment_name>_<timestamp>/
  artifacts/
    experiment_config.<ext>
    experiment_config_resolved.json
    architecture_overview.md
    training_results.json
    train_tokenized/
    validation_tokenized/
  logs/
    debug.log
    train_metrics.jsonl
    eval_metrics.jsonl
```

## Logging for Debugging and Plots

- `debug.log`: human-readable debug and progress logs.
- `train_metrics.jsonl`: raw train metrics per step.
- `eval_metrics.jsonl`: raw eval metrics per step and epoch.

These JSONL files are intended for plotting and post-run analysis.
