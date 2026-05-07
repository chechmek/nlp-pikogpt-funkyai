# DPO Stage

## Purpose

Align an SFT checkpoint further using pairwise preference data and the standard
Direct Preference Optimization objective.

Implementation: `src/dpo/stage.py`

## Data Format

The stage reads local JSONL files with one preference pair per line:

```json
{"prompt":"...", "chosen":"...", "rejected":"..."}
```

The repository includes a local preparation script for `UltraFeedback`:

```bash
python3 scripts/prepare_ultrafeedback_dpo.py \
  --output-dir data/dpo \
  --smoke-samples 500 \
  --train-samples 5000
```

This writes:

- `data/dpo/ultrafeedback_500.jsonl`
- `data/dpo/ultrafeedback_5000.jsonl`

## Run Commands

```bash
# Smoke run on 500 preference pairs
python3 main.py --stage dpo \
  --base-checkpoint runs/models/model_final_sft.pt \
  --dpo-data-path data/dpo/ultrafeedback_500.jsonl \
  --dpo-max-steps 50 \
  --dpo-batch-size 2 \
  --dpo-beta 0.1 \
  --device auto
```

```bash
# Main run on ~5k preference pairs
python3 main.py --stage dpo \
  --base-checkpoint runs/models/model_final_sft.pt \
  --dpo-data-path data/dpo/ultrafeedback_5000.jsonl \
  --dpo-max-steps 200 \
  --dpo-batch-size 2 \
  --dpo-beta 0.1 \
  --device auto
```

## Training Behavior

- The policy model is initialized from `--base-checkpoint`.
- A frozen reference model is created from the same checkpoint.
- Prompt tokens are masked from supervision.
- DPO scores only response-token log-probabilities.
- Final checkpoints are written as `model_final_dpo.pt`.

## Output Structure

Each run creates:

```text
runs_dpo/<run_name>/
  artifacts/
    model_final_dpo.pt
    dpo_results.json
    checkpoints/
  logs/
    dpo_debug.log
    dpo_train_metrics.jsonl
    dpo_eval_metrics.jsonl
```
