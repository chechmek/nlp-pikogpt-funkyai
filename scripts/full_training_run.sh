#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT_DIR/.venv/bin/torchrun}"

MODE="${MODE:-full}"
CONFIG="${CONFIG:-$ROOT_DIR/configs/train_large.toml}"
NUM_SAMPLES="${NUM_SAMPLES:-100000}"
SEED="${SEED:-42}"
TEST_DATA_PATH="${TEST_DATA_PATH:-$ROOT_DIR/data/raw/NLP26_OWT_eval/test}"
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/data/processed/openwebtext_clean}"
SOURCE_DATASET_PATH="${SOURCE_DATASET_PATH:-}"
PROMPT="${PROMPT:-The meaning of life is}"
MAX_TOKENS="${MAX_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-0.8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
RESUME_FROM="${RESUME_FROM:-}"
RUN_NAME="${RUN_NAME:-pipeline_$(date -u +%Y%m%d_%H%M%S)}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python binary not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -x "$TORCHRUN_BIN" ]]; then
  echo "torchrun binary not found or not executable: $TORCHRUN_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"

if [[ "$MODE" == "smoke" ]]; then
  CONFIG="$ROOT_DIR/configs/train_smoke.toml"
  if [[ "$NUM_SAMPLES" == "100000" ]]; then
    NUM_SAMPLES=12
  fi
  SOURCE_DATASET_PATH="$ROOT_DIR/data/smoke/source_dataset"
  TEST_DATA_PATH="$ROOT_DIR/data/smoke/test_dataset"
  OUTPUT_PATH="$ROOT_DIR/data/processed/openwebtext_smoke"
  if [[ "$PROMPT" == "The meaning of life is" ]]; then
    PROMPT="A smoke test prompt"
  fi
  "$PYTHON_BIN" "$ROOT_DIR/scripts/create_smoke_datasets.py" --root "$ROOT_DIR/data/smoke"
fi

if [[ -z "$SOURCE_DATASET_PATH" ]]; then
  echo "SOURCE_DATASET_PATH is required. Preprocessing only supports local datasets on disk." >&2
  exit 1
fi

BASE_DIR="$(
  PYTHONPATH=. "$PYTHON_BIN" - "$CONFIG" "$ROOT_DIR" <<'PY'
from pathlib import Path
import sys
from src.training.config import load_train_config

config_path = Path(sys.argv[1])
root_dir = Path(sys.argv[2])
cfg = load_train_config(config_path)
base_dir = Path(cfg.logging.base_dir)
if not base_dir.is_absolute():
    base_dir = root_dir / base_dir
print(base_dir)
PY
)"

if [[ -z "$SOURCE_DATASET_PATH" ]]; then
  SOURCE_DATASET_PATH=""
fi

export PYTHONPATH="$ROOT_DIR"
export TOKENIZERS_PARALLELISM=false
export PIKOGPT_RUN_NAME="$RUN_NAME"
export MASTER_ADDR
export MASTER_PORT

if [[ "$(uname -s)" == "Darwin" && -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
  export GLOO_SOCKET_IFNAME=lo0
fi

PREPROCESS_CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/main.py"
  --stage preprocess
  --num-samples "$NUM_SAMPLES"
  --seed "$SEED"
  --test-data-path "$TEST_DATA_PATH"
  --output-path "$OUTPUT_PATH"
  --source-dataset-path "$SOURCE_DATASET_PATH"
)

echo "==> Preprocess"
"${PREPROCESS_CMD[@]}"

TRAIN_ARGS=(
  "$ROOT_DIR/main.py"
  --stage train
  --config "$CONFIG"
)

if [[ -n "$RESUME_FROM" ]]; then
  TRAIN_ARGS+=(--resume-from "$RESUME_FROM")
fi

echo "==> Train"
if (( NPROC_PER_NODE > 1 || NNODES > 1 )); then
  if (( NNODES == 1 )); then
    "$TORCHRUN_BIN" \
      --nnodes=1 \
      --nproc_per_node="$NPROC_PER_NODE" \
      --node_rank=0 \
      --master_addr="$MASTER_ADDR" \
      --master_port="$MASTER_PORT" \
      "${TRAIN_ARGS[@]}"
  else
    "$TORCHRUN_BIN" \
      --nnodes="$NNODES" \
      --nproc_per_node="$NPROC_PER_NODE" \
      --node_rank="$NODE_RANK" \
      --master_addr="$MASTER_ADDR" \
      --master_port="$MASTER_PORT" \
      "${TRAIN_ARGS[@]}"
  fi
else
  "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
fi

RUN_DIR="$BASE_DIR/$RUN_NAME"
CHECKPOINT_PATH="$RUN_DIR/artifacts/model_final.pt"
if [[ -n "$RESUME_FROM" ]]; then
  RUN_DIR="$(
    PYTHONPATH=. "$PYTHON_BIN" - "$RESUME_FROM" "$CHECKPOINT_PATH" <<'PY'
from pathlib import Path
import sys

resume_from = Path(sys.argv[1]).resolve()
if resume_from.parent.name == "checkpoints":
    run_dir = resume_from.parent.parent.parent
else:
    run_dir = resume_from.parent.parent
print(run_dir)
PY
  )"
  CHECKPOINT_PATH="$RUN_DIR/artifacts/model_final.pt"
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Expected final checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

echo "==> Inference"
"$PYTHON_BIN" "$ROOT_DIR/main.py" \
  --stage inference \
  --checkpoint "$CHECKPOINT_PATH" \
  --prompt "$PROMPT" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED"

echo
echo "Run directory: $RUN_DIR"
echo "Final checkpoint: $CHECKPOINT_PATH"
