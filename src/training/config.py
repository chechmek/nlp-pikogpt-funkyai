"""
Training configuration models for PikoGPT.

All hyperparameters live in a single TrainStageConfig that is loaded from a
TOML or YAML file. Pydantic validates types and constraints at load time so
bad configs fail early with a clear message rather than mid-training.

Typical usage:
    config = load_train_config("configs/train_large.toml")
    # config.model.n_embd, config.training.learning_rate, etc.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


class ExperimentConfig(BaseModel):
    """Metadata about a training run (name, description, RNG seed)."""

    name: str = "baseline_gpt"
    description: str = "Tokenization + train/eval split + model training setup"
    seed: int = 42


class DataConfig(BaseModel):
    """Paths and split ratios for the pre-processed OpenWebText dataset.

    max_train_samples / max_validation_samples cap the dataset size without
    re-running preprocessing; useful for quick iteration or smoke tests.
    """

    input_dataset_path: str = "data/processed/openwebtext_clean"
    validation_split: float = Field(default=0.1, gt=0.0, lt=0.5)
    split_seed: int = 42
    max_train_samples: int | None = Field(default=None, ge=1)
    max_validation_samples: int | None = Field(default=None, ge=1)


class TokenizerConfig(BaseModel):
    """Tokenizer identity and sequence packing parameters.

    context_length controls the fixed sequence length used during packing —
    every packed sequence will be exactly this many tokens. append_eos_token
    inserts an EOS between documents so the model learns document boundaries.
    """

    name: str = "gpt2"
    context_length: int = Field(default=128, ge=16)
    append_eos_token: bool = True


class ModelConfig(BaseModel):
    """Architecture hyperparameters for CausalTransformerLM.

    vocab_size defaults to None, meaning it is inferred from the tokenizer at
    runtime (50 257 for GPT-2).  n_embd must be divisible by n_head because
    each attention head gets n_embd // n_head dimensions.
    """

    vocab_size: int | None = Field(default=None, ge=1000)
    n_embd: int = Field(default=256, ge=64)
    n_layer: int = Field(default=4, ge=1)
    n_head: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    layer_norm_epsilon: float = Field(default=1e-5, gt=0.0)
    activation: str = "gelu"

    @field_validator("n_head")
    @classmethod
    def validate_num_heads(cls, n_head: int, info):  # type: ignore[override]
        n_embd = info.data.get("n_embd")
        if n_embd is not None and n_embd % n_head != 0:
            raise ValueError("model.n_embd must be divisible by model.n_head")
        return n_head


class TrainingRuntimeConfig(BaseModel):
    """Optimizer, scheduler, and runtime knobs for the training loop.

    grad_accum_steps accumulates gradients over N micro-batches before each
    optimizer step, simulating a larger effective batch without extra memory.
    compile_model runs torch.compile for ~15–20 % throughput gain on CUDA;
    set compile_backend to "eager" to skip compilation on CPU/MPS.
    """

    device: str = "auto"
    batch_size: int = Field(default=8, ge=1)
    eval_batch_size: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    min_learning_rate: float = 1e-5
    weight_decay: float = Field(default=0.01, ge=0.0)
    num_epochs: int = Field(default=1, ge=1)
    max_train_steps: int | None = Field(default=200, ge=1)
    warmup_steps: int = 0  
    grad_accum_steps: int = Field(default=1, ge=1)
    log_every_steps: int = Field(default=10, ge=1)
    eval_every_steps: int = Field(default=50, ge=1)
    gradient_clip_norm: float | None = Field(default=1.0, gt=0.0)
    compile_model: bool = True
    compile_backend: str = "auto"
    compile_mode: str | None = "default"
    save_every_steps: int | None = Field(default=500, ge=1)
    resume_from_checkpoint: str | None = None


class LoggingConfig(BaseModel):
    """Output directory and JSONL filenames for training metrics.

    All run artefacts (checkpoints, logs, config snapshot) are written under
    base_dir/<run_name>/.  The JSONL files record per-step and per-epoch
    metrics for later plotting with scripts/plot_training_metrics.py.
    """

    base_dir: str = "runs"
    level: str = "INFO"
    train_jsonl_name: str = "train_metrics.jsonl"
    eval_jsonl_name: str = "eval_metrics.jsonl"

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, level: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = level.upper()
        if normalized not in allowed:
            raise ValueError(f"logging.level must be one of {sorted(allowed)}")
        return normalized


class TrainStageConfig(BaseModel):
    """Root config that composes all sub-configs for a full training run.

    Loaded from a TOML/YAML file via load_train_config().  Each section maps
    to a sub-config class: [experiment], [data], [tokenizer], [model],
    [training], [logging].
    """

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingRuntimeConfig = Field(default_factory=TrainingRuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _load_raw_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("rb") as f:
        if suffix == ".toml":
            raw = tomllib.load(f)
        elif suffix in {".yaml", ".yml"}:
            raw = yaml.safe_load(f)
        else:
            raise ValueError("Config must be a .toml, .yaml, or .yml file")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping/object")
    return raw


def model_dump_compat(model: BaseModel) -> dict[str, Any]:
    """Serialize a Pydantic model to a plain dict regardless of v1/v2."""
    if hasattr(model, "model_dump"):
        return model.model_dump()  # pydantic v2
    return model.dict()  # pragma: no cover - pydantic v1


def load_train_config(config_path: str | Path) -> TrainStageConfig:
    """Load and validate a training config from a TOML or YAML file.

    Raises FileNotFoundError if the path does not exist and ValueError for
    unsupported extensions or invalid TOML/YAML structure.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = _load_raw_config(path)
    if hasattr(TrainStageConfig, "model_validate"):
        return TrainStageConfig.model_validate(raw)  # pydantic v2
    return TrainStageConfig.parse_obj(raw)  # pragma: no cover - pydantic v1
