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
  --config configs/train_default.toml
```

```bash
# Prepare-only mode (debug pipeline, no optimizer steps)
PYTHONPATH=. .venv/bin/python main.py \
  --stage train \
  --config configs/train_default.toml \
  --prepare-only
```

```bash
# Resume from a periodic checkpoint
PYTHONPATH=. .venv/bin/python main.py \
  --stage train \
  --config configs/train_large.toml \
  --resume-from runs/<run_name>/artifacts/checkpoints/step_005000.pt
```

```bash
# Single-node distributed run
PYTHONPATH=. .venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  main.py --stage train --config configs/train_large.toml
```

```bash
# End-to-end pipeline (preprocess -> train -> inference)
MODE=smoke scripts/full_training_run.sh
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

Notable training options:

- `training.compile_model`: enables `torch.compile`.
- `training.compile_backend`: `auto` selects `inductor` on CUDA and `eager` elsewhere.
- `training.save_every_steps`: writes resumable checkpoints under `artifacts/checkpoints/`.
- `training.resume_from_checkpoint`: optional checkpoint path for resume.

## Output Structure

Each run creates:

```text
runs/<experiment_name>_<timestamp>/
  artifacts/
    experiment_config.<ext>
    experiment_config_resolved.json
    architecture_overview.md
    training_results.json
    model_final.pt
    checkpoints/
    train_tokenized/
    validation_tokenized/
  logs/
    debug_rank0.log
    train_metrics.jsonl
    eval_metrics.jsonl
```

## Logging for Debugging and Plots

- `debug.log`: human-readable debug and progress logs.
- `train_metrics.jsonl`: raw train metrics per step.
- `eval_metrics.jsonl`: raw eval metrics per step and epoch.

These JSONL files are intended for plotting and post-run analysis.
