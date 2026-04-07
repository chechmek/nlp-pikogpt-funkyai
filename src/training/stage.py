from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from datasets import Dataset, load_from_disk
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from .config import TrainStageConfig, load_train_config, model_dump_compat
from .utils import (
    cleanup_distributed,
    compute_gradient_norm,
    get_cosine_schedule_with_warmup,
    is_main_process,
    print_model_summary,
    print_training_config,
    setup_distributed,
)


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def distributed_barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def broadcast_object(value: Any) -> Any:
    if not dist.is_initialized():
        return value
    payload = [value if get_rank() == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def reduce_mean(value: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float((tensor / get_world_size()).item())


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    requested = device_name.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dict")

    state_dict = payload.get("state_dict") or payload.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint is missing model weights")

    return payload


def resolve_run_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent.parent
    return checkpoint_path.parent.parent


class CausalTransformerLM(nn.Module):
    """
    Decoder-only language model implemented with nn.TransformerEncoder + causal mask.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        dropout: float,
        layer_norm_epsilon: float,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(max_seq_len, n_embd)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation="gelu",
            layer_norm_eps=layer_norm_epsilon,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.final_norm = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model max_seq_len {self.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = self._build_causal_mask(seq_len=seq_len, device=input_ids.device)
        x = self.transformer(x, mask=causal_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss: torch.Tensor | None = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {"logits": logits, "loss": loss}


class TrainStage:
    def __init__(
        self,
        config_path: str | Path,
        prepare_only: bool = False,
        resume_from_checkpoint: str | Path | None = None,
    ):
        self.config_path = Path(config_path)
        self.prepare_only = prepare_only
        self.resume_from_override = Path(resume_from_checkpoint) if resume_from_checkpoint else None

        self.config: TrainStageConfig | None = None
        self.run_dir: Path | None = None
        self.artifacts_dir: Path | None = None
        self.logs_dir: Path | None = None
        self.checkpoints_dir: Path | None = None

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.use_ddp = self.world_size > 1
        self.resume_checkpoint_path: Path | None = None

        self.logger: logging.Logger | None = None
        self.train_jsonl: JsonlWriter | None = None
        self.eval_jsonl: JsonlWriter | None = None

    def run(self) -> dict[str, Any]:
        started_at = time.time()
        self._initialize()

        self.logger.info("Starting train stage")
        self.logger.info("Prepare-only mode: %s", self.prepare_only)
        self.logger.info("Distributed mode: %s (world_size=%s)", self.use_ddp, self.world_size)
        if self.resume_checkpoint_path is not None:
            self.logger.info("Resuming from checkpoint: %s", self.resume_checkpoint_path)

        tokenizer, tokenized_train, tokenized_eval = self._prepare_datasets()

        model = self._build_model(tokenizer)
        if self.resume_checkpoint_path is not None:
            checkpoint_payload = load_checkpoint_payload(self.resume_checkpoint_path)
            model.load_state_dict(checkpoint_payload["state_dict"])
        else:
            checkpoint_payload = None

        if is_main_process():
            self._write_architecture_overview(model, tokenizer)

        if self.prepare_only:
            results = {
                "status": "prepared_only",
                "timestamp": utc_now_iso(),
                "train_sequences": len(tokenized_train),
                "validation_sequences": len(tokenized_eval),
                "world_size": self.world_size,
                "duration_seconds": round(time.time() - started_at, 2),
            }
            if is_main_process():
                self._write_results(results)
                self.logger.info("Preparation completed without running optimizer steps")
            distributed_barrier()
            return results

        training_results = self._train_model(
            base_model=model,
            tokenizer=tokenizer,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            checkpoint_payload=checkpoint_payload,
        )

        if is_main_process():
            checkpoint_path = self._save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=None,
                scheduler=None,
                training_state=training_results,
                filename="model_final.pt",
                checkpoint_kind="final",
            )
            training_results["status"] = "completed"
            training_results["checkpoint_path"] = str(checkpoint_path)
            training_results["duration_seconds"] = round(time.time() - started_at, 2)
            self._write_results(training_results)
            self.logger.info("Training completed")

        distributed_barrier()
        return training_results

    def _initialize(self) -> None:
        self.config = load_train_config(self.config_path)
        if self.resume_from_override is not None:
            self.config.training.resume_from_checkpoint = str(self.resume_from_override)

        self.resume_checkpoint_path = None
        if self.config.training.resume_from_checkpoint:
            self.resume_checkpoint_path = Path(self.config.training.resume_from_checkpoint).expanduser().resolve()
            if not self.resume_checkpoint_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {self.resume_checkpoint_path}")
            self.run_dir = resolve_run_dir_from_checkpoint(self.resume_checkpoint_path)
        else:
            self.run_dir = Path(self.config.logging.base_dir) / self._resolve_run_name()

        self.artifacts_dir = self.run_dir / "artifacts"
        self.logs_dir = self.run_dir / "logs"
        self.checkpoints_dir = self.artifacts_dir / "checkpoints"

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._build_logger()
        self.train_jsonl = JsonlWriter(self.logs_dir / self.config.logging.train_jsonl_name)
        self.eval_jsonl = JsonlWriter(self.logs_dir / self.config.logging.eval_jsonl_name)

        set_seed(self.config.experiment.seed + self.rank)
        self._snapshot_config()
        distributed_barrier()

    def _resolve_run_name(self) -> str:
        env_name = os.environ.get("PIKOGPT_RUN_NAME")
        if env_name:
            return env_name

        generated_name = None
        if not self.use_ddp or self.rank == 0:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            generated_name = f"{self.config.experiment.name}_{timestamp}"
        return str(broadcast_object(generated_name))

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"train_stage.{self.run_dir.name}.rank{self.rank}")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.propagate = False

        formatter = logging.Formatter(f"%(asctime)s | rank={self.rank} | %(levelname)s | %(message)s")

        if is_main_process():
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.config.logging.level))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        file_handler = logging.FileHandler(
            self.logs_dir / f"debug_rank{self.rank}.log",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def _snapshot_config(self) -> None:
        if not is_main_process():
            return

        if self.resume_checkpoint_path is None:
            raw_config_copy = self.artifacts_dir / f"experiment_config{self.config_path.suffix.lower()}"
            resolved_path = self.artifacts_dir / "experiment_config_resolved.json"
        else:
            raw_config_copy = self.artifacts_dir / f"resume_config{self.config_path.suffix.lower()}"
            resolved_path = self.artifacts_dir / "resume_config_resolved.json"

        shutil.copy2(self.config_path, raw_config_copy)
        resolved_path.write_text(
            json.dumps(model_dump_compat(self.config), indent=2),
            encoding="utf-8",
        )

    def _load_raw_dataset(self) -> Dataset:
        dataset_path = Path(self.config.data.input_dataset_path)
        self.logger.info("Loading processed dataset from: %s", dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {dataset_path}. "
                "Run preprocess first, for example: "
                "python main.py --stage preprocess --output-path data/processed/openwebtext_clean"
            )

        dataset = load_from_disk(str(dataset_path))

        if "text" not in dataset.column_names:
            raise ValueError(
                f"Expected a 'text' column in dataset. Found: {dataset.column_names}"
            )

        self.logger.info("Loaded %s documents", f"{len(dataset):,}")
        return dataset

    def _split_dataset(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        split = dataset.train_test_split(
            test_size=self.config.data.validation_split,
            seed=self.config.data.split_seed,
            shuffle=True,
        )
        train_text = split["train"]
        eval_text = split["test"]

        if self.config.data.max_train_samples is not None:
            train_limit = min(len(train_text), self.config.data.max_train_samples)
            train_text = train_text.select(range(train_limit))

        if self.config.data.max_validation_samples is not None:
            eval_limit = min(len(eval_text), self.config.data.max_validation_samples)
            eval_text = eval_text.select(range(eval_limit))

        if len(train_text) == 0 or len(eval_text) == 0:
            raise ValueError("Train/validation split produced an empty dataset")

        self.logger.info(
            "Split dataset -> train docs: %s | validation docs: %s",
            f"{len(train_text):,}",
            f"{len(eval_text):,}",
        )
        return train_text, eval_text

    def _create_tokenizer(self):
        self.logger.info("Loading tokenizer: %s", self.config.tokenizer.name)
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer.name, use_fast=True)
        tokenizer.model_max_length = 10_000_000

        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no eos_token_id; cannot build causal LM sequences")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _tokenize_and_pack(self, text_dataset: Dataset, tokenizer, split_name: str) -> Dataset:
        self.logger.info("Tokenizing %s split (%s docs)", split_name, f"{len(text_dataset):,}")

        def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
            encoded = tokenizer(batch["text"], add_special_tokens=False)
            return {"input_ids": encoded["input_ids"]}

        tokenized = text_dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=text_dataset.column_names,
            desc=f"Tokenizing {split_name}",
        )

        block_size = self.config.tokenizer.context_length
        eos_id = tokenizer.eos_token_id
        append_eos = self.config.tokenizer.append_eos_token and eos_id is not None

        def pack_batch(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
            flattened: list[int] = []
            for token_ids in batch["input_ids"]:
                flattened.extend(token_ids)
                if append_eos:
                    flattened.append(eos_id)

            usable_tokens = (len(flattened) // block_size) * block_size
            if usable_tokens == 0:
                return {"input_ids": [], "labels": []}

            packed = [
                flattened[i : i + block_size]
                for i in range(0, usable_tokens, block_size)
            ]
            return {
                "input_ids": packed,
                "labels": [chunk.copy() for chunk in packed],
            }

        packed_dataset = tokenized.map(
            pack_batch,
            batched=True,
            remove_columns=tokenized.column_names,
            desc=f"Packing {split_name}",
        )

        if len(packed_dataset) == 0:
            raise ValueError(
                f"Tokenization produced 0 sequences for split '{split_name}'. "
                "Try lowering tokenizer.context_length or increasing data size."
            )

        self.logger.info(
            "Packed %s split into %s sequences of length %s",
            split_name,
            f"{len(packed_dataset):,}",
            block_size,
        )
        return packed_dataset

    def _tokenized_dataset_paths(self) -> tuple[Path, Path]:
        return self.artifacts_dir / "train_tokenized", self.artifacts_dir / "validation_tokenized"

    def _prepare_datasets(self):
        train_path, eval_path = self._tokenized_dataset_paths()
        datasets_exist = train_path.exists() and eval_path.exists()

        if not datasets_exist:
            if is_main_process():
                dataset = self._load_raw_dataset()
                train_text, eval_text = self._split_dataset(dataset)
                prep_tokenizer = self._create_tokenizer()
                tokenized_train = self._tokenize_and_pack(train_text, prep_tokenizer, split_name="train")
                tokenized_eval = self._tokenize_and_pack(eval_text, prep_tokenizer, split_name="validation")
                self._save_tokenized_datasets(tokenized_train, tokenized_eval)
            else:
                self.logger.info("Waiting for rank 0 to prepare tokenized datasets")

        distributed_barrier()

        tokenizer = self._create_tokenizer()
        tokenized_train = load_from_disk(str(train_path))
        tokenized_eval = load_from_disk(str(eval_path))
        self.logger.info(
            "Loaded tokenized datasets -> train sequences: %s | validation sequences: %s",
            f"{len(tokenized_train):,}",
            f"{len(tokenized_eval):,}",
        )
        return tokenizer, tokenized_train, tokenized_eval

    def _save_tokenized_datasets(self, train_dataset: Dataset, eval_dataset: Dataset) -> None:
        train_path, eval_path = self._tokenized_dataset_paths()
        for path in (train_path, eval_path):
            if path.exists():
                shutil.rmtree(path)

        train_dataset.save_to_disk(str(train_path))
        eval_dataset.save_to_disk(str(eval_path))

        self.logger.info("Saved tokenized train split to: %s", train_path)
        self.logger.info("Saved tokenized validation split to: %s", eval_path)

    def _build_model(self, tokenizer) -> CausalTransformerLM:
        vocab_size = self.config.model.vocab_size or tokenizer.vocab_size

        return CausalTransformerLM(
            vocab_size=vocab_size,
            max_seq_len=self.config.tokenizer.context_length,
            n_embd=self.config.model.n_embd,
            n_layer=self.config.model.n_layer,
            n_head=self.config.model.n_head,
            dropout=self.config.model.dropout,
            layer_norm_epsilon=self.config.model.layer_norm_epsilon,
        )

    def _write_architecture_overview(self, model: CausalTransformerLM, tokenizer) -> None:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        overview = [
            "# Architecture Overview",
            "",
            f"- Run name: `{self.run_dir.name}`",
            f"- Tokenizer: `{self.config.tokenizer.name}`",
            f"- Context length: `{self.config.tokenizer.context_length}`",
            f"- Vocabulary size: `{tokenizer.vocab_size}`",
            "",
            "## Decoder-Only Transformer Definition",
            "- Backend: `torch.nn.TransformerEncoder` with causal self-attention mask",
            f"- Layers (`n_layer`): `{self.config.model.n_layer}`",
            f"- Heads (`n_head`): `{self.config.model.n_head}`",
            f"- Embedding dimension (`n_embd`): `{self.config.model.n_embd}`",
            f"- Dropout: `{self.config.model.dropout}`",
            "",
            "## Parameter Count",
            f"- Total parameters: `{total_params:,}`",
            f"- Trainable parameters: `{trainable_params:,}`",
            "",
            "## Training Definition",
            f"- Epochs: `{self.config.training.num_epochs}`",
            f"- Batch size: `{self.config.training.batch_size}`",
            f"- Eval batch size: `{self.config.training.eval_batch_size}`",
            f"- Learning rate: `{self.config.training.learning_rate}`",
            f"- Weight decay: `{self.config.training.weight_decay}`",
            f"- Max train steps: `{self.config.training.max_train_steps}`",
            f"- torch.compile: `{self.config.training.compile_model}`",
            f"- Save every steps: `{self.config.training.save_every_steps}`",
        ]

        overview_path = self.artifacts_dir / "architecture_overview.md"
        overview_path.write_text("\n".join(overview), encoding="utf-8")
        self.logger.info("Saved architecture overview to: %s", overview_path)

    def _resolve_training_device(self) -> tuple[torch.device, dict[str, Any]]:
        if self.use_ddp and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.local_rank)
            return device, {"device_ids": [self.local_rank], "output_device": self.local_rank}
        if self.use_ddp:
            return torch.device("cpu"), {}
        return resolve_device(self.config.training.device), {}

    def _compile_model(self, model: nn.Module, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
        compile_requested = bool(self.config.training.compile_model)
        if not compile_requested:
            return model, {"enabled": False}
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this torch build")

        backend = self.config.training.compile_backend
        if backend == "auto":
            backend = "inductor" if device.type == "cuda" else "eager"

        kwargs: dict[str, Any] = {"backend": backend}
        if self.config.training.compile_mode is not None:
            kwargs["mode"] = self.config.training.compile_mode

        compiled = torch.compile(model, **kwargs)
        return compiled, {
            "enabled": True,
            "backend": backend,
            "mode": self.config.training.compile_mode,
        }

    def _create_checkpoint_payload(
        self,
        model: CausalTransformerLM,
        tokenizer,
        optimizer,
        scheduler,
        training_state: dict[str, Any] | None,
        checkpoint_kind: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": "pikogpt_checkpoint_v2",
            "created_at": utc_now_iso(),
            "checkpoint_kind": checkpoint_kind,
            "model": {
                "vocab_size": model.vocab_size,
                "max_seq_len": model.max_seq_len,
                "n_embd": self.config.model.n_embd,
                "n_layer": self.config.model.n_layer,
                "n_head": self.config.model.n_head,
                "dropout": self.config.model.dropout,
                "layer_norm_epsilon": self.config.model.layer_norm_epsilon,
            },
            "tokenizer": {
                "name": getattr(tokenizer, "name_or_path", self.config.tokenizer.name),
                "context_length": self.config.tokenizer.context_length,
                "append_eos_token": self.config.tokenizer.append_eos_token,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            },
            "state_dict": {
                key: tensor.detach().cpu()
                for key, tensor in model.state_dict().items()
            },
            "config": model_dump_compat(self.config),
            "run_dir": str(self.run_dir),
        }

        if training_state is not None:
            payload["training_state"] = training_state
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()

        return payload

    def _save_checkpoint(
        self,
        model: CausalTransformerLM,
        tokenizer,
        optimizer,
        scheduler,
        training_state: dict[str, Any] | None,
        filename: str,
        checkpoint_kind: str,
    ) -> Path:
        checkpoint_path = self.artifacts_dir / filename
        payload = self._create_checkpoint_payload(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            checkpoint_kind=checkpoint_kind,
        )
        torch.save(payload, checkpoint_path)
        self.logger.info("Saved %s checkpoint to: %s", checkpoint_kind, checkpoint_path)
        return checkpoint_path

    def _save_step_checkpoint(
        self,
        model: CausalTransformerLM,
        tokenizer,
        optimizer,
        scheduler,
        training_state: dict[str, Any],
    ) -> Path:
        checkpoint_path = self.checkpoints_dir / f"step_{training_state['global_step']:06d}.pt"
        payload = self._create_checkpoint_payload(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            checkpoint_kind="step",
        )
        torch.save(payload, checkpoint_path)
        self.logger.info("Saved step checkpoint to: %s", checkpoint_path)
        return checkpoint_path

    def _build_resume_state(self, checkpoint_payload: dict[str, Any] | None) -> dict[str, Any]:
        if checkpoint_payload is None:
            return {
                "global_step": 0,
                "micro_step": 0,
                "epoch": 1,
                "batch_idx": 0,
                "best_eval_loss": None,
                "epoch_metrics": [],
            }

        training_state = checkpoint_payload.get("training_state")
        if not isinstance(training_state, dict):
            self.logger.warning("Checkpoint has no resumable training_state; weights will load but optimizer state will restart")
            return {
                "global_step": 0,
                "micro_step": 0,
                "epoch": 1,
                "batch_idx": 0,
                "best_eval_loss": None,
                "epoch_metrics": [],
            }

        return {
            "global_step": int(training_state.get("global_step", 0)),
            "micro_step": int(training_state.get("micro_step", 0)),
            "epoch": int(training_state.get("epoch", 1)),
            "batch_idx": int(training_state.get("batch_idx", 0)),
            "best_eval_loss": training_state.get("best_eval_loss"),
            "epoch_metrics": list(training_state.get("epoch_metrics", [])),
        }

    def _train_model(
        self,
        base_model: CausalTransformerLM,
        tokenizer,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        checkpoint_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        device, ddp_kwargs = self._resolve_training_device()
        base_model.to(device)

        optimizer = AdamW(
            base_model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        train_torch = train_dataset.with_format("torch")
        eval_torch = eval_dataset.with_format("torch")

        train_sampler = (
            DistributedSampler(train_torch, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            if self.use_ddp
            else None
        )
        eval_sampler = (
            DistributedSampler(eval_torch, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            if self.use_ddp
            else None
        )

        train_loader = DataLoader(
            train_torch,
            batch_size=self.config.training.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=len(train_torch) >= self.config.training.batch_size,
        )
        eval_loader = DataLoader(
            eval_torch,
            batch_size=self.config.training.eval_batch_size,
            shuffle=False,
            sampler=eval_sampler,
        )

        if len(train_loader) == 0:
            raise ValueError("Training dataloader is empty; reduce batch size or increase training data")

        grad_accum_steps = self.config.training.grad_accum_steps
        if self.config.training.max_train_steps is not None:
            total_steps = self.config.training.max_train_steps
        else:
            steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
            total_steps = steps_per_epoch * self.config.training.num_epochs

        warmup_steps = getattr(self.config.training, "warmup_steps", 0)
        min_lr = getattr(self.config.training, "min_learning_rate", 1e-5)
        min_lr_ratio = min_lr / self.config.training.learning_rate
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )

        resume_state = self._build_resume_state(checkpoint_payload)
        if checkpoint_payload is not None and checkpoint_payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
        if checkpoint_payload is not None and checkpoint_payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint_payload["scheduler_state_dict"])

        compiled_model, compile_details = self._compile_model(base_model, device)
        model = DDP(compiled_model, **ddp_kwargs) if self.use_ddp else compiled_model

        if is_main_process():
            print_model_summary(base_model, self.config, self.logger)
            print_training_config(self.config, total_steps, self.logger)
            eff_batch = self.config.training.batch_size * self.world_size * grad_accum_steps
            self.logger.info(
                "Effective batch size: %s (micro=%s x world=%s x accum=%s)",
                eff_batch,
                self.config.training.batch_size,
                self.world_size,
                grad_accum_steps,
            )
            self.logger.info("Using device: %s", device)
            self.logger.info("torch.compile details: %s", compile_details)
            if self.resume_checkpoint_path is not None:
                self.logger.info(
                    "Resume state -> epoch=%s batch_idx=%s global_step=%s",
                    resume_state["epoch"],
                    resume_state["batch_idx"],
                    resume_state["global_step"],
                )

        global_step = resume_state["global_step"]
        micro_step = resume_state["micro_step"]
        best_eval_loss = resume_state["best_eval_loss"]
        epoch_results: list[dict[str, Any]] = resume_state["epoch_metrics"]
        training_started = time.time()
        stop_training = global_step >= total_steps

        if stop_training and is_main_process():
            self.logger.info("Checkpoint already reached max_train_steps=%s; skipping optimizer loop", total_steps)

        start_epoch = resume_state["epoch"]
        resume_batch_idx = resume_state["batch_idx"]

        for epoch in range(start_epoch, self.config.training.num_epochs + 1):
            if stop_training:
                break

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if eval_sampler is not None:
                eval_sampler.set_epoch(epoch)

            if epoch == start_epoch and resume_batch_idx >= len(train_loader):
                resume_batch_idx = 0
                continue

            model.train()
            step_losses: list[float] = []
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for batch_idx, batch in enumerate(train_loader, start=1):
                if epoch == start_epoch and batch_idx <= resume_batch_idx:
                    continue

                micro_step += 1
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                is_sync_step = (
                    batch_idx % grad_accum_steps == 0
                ) or (batch_idx == len(train_loader))

                if self.use_ddp and not is_sync_step:
                    with model.no_sync():
                        outputs = model(input_ids=input_ids, labels=labels)
                        loss = outputs["loss"]
                        if loss is None:
                            raise RuntimeError("Model returned no loss")
                        (loss / grad_accum_steps).backward()
                else:
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs["loss"]
                    if loss is None:
                        raise RuntimeError("Model returned no loss")
                    (loss / grad_accum_steps).backward()

                accum_loss += float(loss.detach().item())

                if is_sync_step:
                    grad_norm = compute_gradient_norm(base_model)

                    if self.config.training.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            base_model.parameters(),
                            max_norm=self.config.training.gradient_clip_norm,
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    avg_loss = accum_loss / grad_accum_steps
                    avg_loss = reduce_mean(avg_loss, device)
                    step_losses.append(avg_loss)
                    current_lr = scheduler.get_last_lr()[0]

                    if is_main_process():
                        self.train_jsonl.write(
                            {
                                "timestamp": utc_now_iso(),
                                "event": "train_step",
                                "epoch": epoch,
                                "batch_idx": batch_idx,
                                "step": global_step,
                                "loss": avg_loss,
                                "lr": current_lr,
                                "grad_norm": grad_norm,
                            }
                        )
                        if global_step % self.config.training.log_every_steps == 0:
                            self.logger.info(
                                "Epoch %s | Step %s | Batch %s | Train loss %.4f | lr %.2e | grad %.4f",
                                epoch,
                                global_step,
                                batch_idx,
                                avg_loss,
                                current_lr,
                                grad_norm,
                            )

                    if global_step % self.config.training.eval_every_steps == 0:
                        eval_loss = self._evaluate(model, eval_loader, device)
                        if is_main_process():
                            self.eval_jsonl.write(
                                {
                                    "timestamp": utc_now_iso(),
                                    "event": "eval_step",
                                    "epoch": epoch,
                                    "batch_idx": batch_idx,
                                    "step": global_step,
                                    "eval_loss": eval_loss,
                                }
                            )
                            self.logger.info(
                                "Epoch %s | Step %s | Eval loss %s",
                                epoch,
                                global_step,
                                f"{eval_loss:.4f}" if eval_loss is not None else "n/a",
                            )
                        if eval_loss is not None and (
                            best_eval_loss is None or eval_loss < best_eval_loss
                        ):
                            best_eval_loss = eval_loss

                    if (
                        is_main_process()
                        and self.config.training.save_every_steps is not None
                        and global_step % self.config.training.save_every_steps == 0
                        and global_step < total_steps
                    ):
                        checkpoint_state = {
                            "timestamp": utc_now_iso(),
                            "global_step": global_step,
                            "micro_step": micro_step,
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "best_eval_loss": best_eval_loss,
                            "epoch_metrics": epoch_results,
                            "world_size": self.world_size,
                            "compile": compile_details,
                            "device": str(device),
                        }
                        self._save_step_checkpoint(
                            model=base_model,
                            tokenizer=tokenizer,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            training_state=checkpoint_state,
                        )

                    if global_step >= total_steps:
                        stop_training = True
                        accum_loss = 0.0
                        break

                    accum_loss = 0.0

            if stop_training and not step_losses and epoch == start_epoch:
                break

            epoch_train_loss = sum(step_losses) / len(step_losses) if step_losses else None
            epoch_eval_loss = self._evaluate(model, eval_loader, device)

            if epoch_eval_loss is not None and (
                best_eval_loss is None or epoch_eval_loss < best_eval_loss
            ):
                best_eval_loss = epoch_eval_loss

            epoch_payload = {
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "eval_loss": epoch_eval_loss,
            }
            epoch_results.append(epoch_payload)

            if is_main_process():
                self.eval_jsonl.write(
                    {
                        "timestamp": utc_now_iso(),
                        "event": "epoch_end",
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": epoch_train_loss,
                        "eval_loss": epoch_eval_loss,
                    }
                )
                self.logger.info(
                    "Epoch %s complete | train_loss=%s | eval_loss=%s",
                    epoch,
                    f"{epoch_train_loss:.4f}" if epoch_train_loss is not None else "n/a",
                    f"{epoch_eval_loss:.4f}" if epoch_eval_loss is not None else "n/a",
                )

            resume_batch_idx = 0

        total_params = sum(p.numel() for p in base_model.parameters())

        return {
            "timestamp": utc_now_iso(),
            "device": str(device),
            "global_step": global_step,
            "global_steps": global_step,
            "micro_step": micro_step,
            "epoch": epoch_results[-1]["epoch"] if epoch_results else start_epoch,
            "batch_idx": 0,
            "epochs_completed": len(epoch_results),
            "best_eval_loss": best_eval_loss,
            "epoch_metrics": epoch_results,
            "num_parameters": total_params,
            "training_seconds": round(time.time() - training_started, 2),
            "train_sequences": len(train_dataset),
            "validation_sequences": len(eval_dataset),
            "world_size": self.world_size,
            "grad_accum_steps": grad_accum_steps,
            "compile": compile_details,
            "resumed_from": str(self.resume_checkpoint_path) if self.resume_checkpoint_path else None,
        }

    def _evaluate(
        self,
        model: nn.Module,
        eval_loader: DataLoader,
        device: torch.device,
    ) -> float | None:
        model.eval()
        loss_sum = 0.0
        batch_count = 0.0

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]
                if loss is None:
                    continue
                loss_sum += float(loss.detach().item())
                batch_count += 1.0

        if self.use_ddp:
            stats = torch.tensor([loss_sum, batch_count], dtype=torch.float64, device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            loss_sum = float(stats[0].item())
            batch_count = float(stats[1].item())

        model.train()

        if batch_count == 0:
            return None
        return loss_sum / batch_count

    def _write_results(self, results: dict[str, Any]) -> None:
        payload = dict(results)
        payload["run_dir"] = str(self.run_dir)
        payload["artifacts_dir"] = str(self.artifacts_dir)
        payload["logs_dir"] = str(self.logs_dir)

        results_path = self.artifacts_dir / "training_results.json"
        results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.logger.info("Saved training results to: %s", results_path)


def main(
    config_path: str,
    prepare_only: bool = False,
    resume_from_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    use_ddp = "RANK" in os.environ and "LOCAL_RANK" in os.environ

    if use_ddp:
        setup_distributed()

    try:
        stage = TrainStage(
            config_path=config_path,
            prepare_only=prepare_only,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        return stage.run()
    finally:
        if use_ddp:
            cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PikoGPT train stage")
    parser.add_argument("--config", required=True, help="Path to .toml/.yaml config")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run tokenization/splitting/log setup only, skip optimizer training",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume training from a checkpoint path",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        prepare_only=args.prepare_only,
        resume_from_checkpoint=args.resume_from,
    )
