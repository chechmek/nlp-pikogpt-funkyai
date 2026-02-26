from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_from_disk
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import TrainStageConfig, load_train_config, model_dump_compat


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        # Weight tying: improves efficiency and is standard in GPT-style LMs.
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
    def __init__(self, config_path: str | Path, prepare_only: bool = False):
        self.config_path = Path(config_path)
        self.prepare_only = prepare_only

        self.config: TrainStageConfig | None = None
        self.run_dir: Path | None = None
        self.artifacts_dir: Path | None = None
        self.logs_dir: Path | None = None

        self.logger: logging.Logger | None = None
        self.train_jsonl: JsonlWriter | None = None
        self.eval_jsonl: JsonlWriter | None = None

    def run(self) -> dict[str, Any]:
        started_at = time.time()
        self._initialize()

        self.logger.info("Starting train stage")
        self.logger.info("Prepare-only mode: %s", self.prepare_only)

        dataset = self._load_raw_dataset()
        train_text, eval_text = self._split_dataset(dataset)

        tokenizer = self._create_tokenizer()
        tokenized_train = self._tokenize_and_pack(train_text, tokenizer, split_name="train")
        tokenized_eval = self._tokenize_and_pack(eval_text, tokenizer, split_name="validation")

        self._save_tokenized_datasets(tokenized_train, tokenized_eval)

        model = self._build_model(tokenizer)
        self._write_architecture_overview(model, tokenizer)

        if self.prepare_only:
            results = {
                "status": "prepared_only",
                "timestamp": utc_now_iso(),
                "train_sequences": len(tokenized_train),
                "validation_sequences": len(tokenized_eval),
                "duration_seconds": round(time.time() - started_at, 2),
            }
            self._write_results(results)
            self.logger.info("Preparation completed without running optimizer steps")
            return results

        training_results = self._train_model(model, tokenized_train, tokenized_eval)
        training_results["status"] = "completed"
        training_results["duration_seconds"] = round(time.time() - started_at, 2)

        self._write_results(training_results)
        self.logger.info("Training completed")
        return training_results

    def _initialize(self) -> None:
        self.config = load_train_config(self.config_path)
        set_seed(self.config.experiment.seed)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.config.experiment.name}_{timestamp}"

        self.run_dir = Path(self.config.logging.base_dir) / run_name
        self.artifacts_dir = self.run_dir / "artifacts"
        self.logs_dir = self.run_dir / "logs"

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._build_logger()
        self.train_jsonl = JsonlWriter(self.logs_dir / self.config.logging.train_jsonl_name)
        self.eval_jsonl = JsonlWriter(self.logs_dir / self.config.logging.eval_jsonl_name)

        self._snapshot_config()

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"train_stage.{self.run_dir.name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.propagate = False

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.logging.level))
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(self.logs_dir / "debug.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        return logger

    def _snapshot_config(self) -> None:
        raw_config_copy = self.artifacts_dir / f"experiment_config{self.config_path.suffix.lower()}"
        shutil.copy2(self.config_path, raw_config_copy)

        resolved_path = self.artifacts_dir / "experiment_config_resolved.json"
        resolved_path.write_text(
            json.dumps(model_dump_compat(self.config), indent=2),
            encoding="utf-8",
        )

    def _load_raw_dataset(self) -> Dataset:
        self.logger.info("Loading processed dataset from: %s", self.config.data.input_dataset_path)
        dataset = load_from_disk(self.config.data.input_dataset_path)

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
        # We pack tokens manually into fixed blocks, so we disable tokenizer max-length warnings.
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

    def _save_tokenized_datasets(self, train_dataset: Dataset, eval_dataset: Dataset) -> None:
        train_path = self.artifacts_dir / "train_tokenized"
        eval_path = self.artifacts_dir / "validation_tokenized"

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
        ]

        overview_path = self.artifacts_dir / "architecture_overview.md"
        overview_path.write_text("\n".join(overview), encoding="utf-8")
        self.logger.info("Saved architecture overview to: %s", overview_path)

    def _train_model(
        self,
        model: CausalTransformerLM,
        train_dataset: Dataset,
        eval_dataset: Dataset,
    ) -> dict[str, Any]:
        device = resolve_device(self.config.training.device)
        model.to(device)

        self.logger.info("Using device: %s", device)

        train_torch = train_dataset.with_format("torch")
        eval_torch = eval_dataset.with_format("torch")

        train_loader = DataLoader(
            train_torch,
            batch_size=self.config.training.batch_size,
            shuffle=True,
        )
        eval_loader = DataLoader(
            eval_torch,
            batch_size=self.config.training.eval_batch_size,
            shuffle=False,
        )

        optimizer = AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        global_step = 0
        best_eval_loss: float | None = None
        epoch_results: list[dict[str, Any]] = []
        training_started = time.time()

        stop_training = False
        for epoch in range(1, self.config.training.num_epochs + 1):
            model.train()
            step_losses: list[float] = []

            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]
                if loss is None:
                    raise RuntimeError("Model returned no loss while labels were provided")
                loss.backward()

                if self.config.training.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=self.config.training.gradient_clip_norm,
                    )

                optimizer.step()

                global_step += 1
                loss_value = float(loss.detach().cpu().item())
                step_losses.append(loss_value)

                self.train_jsonl.write(
                    {
                        "timestamp": utc_now_iso(),
                        "event": "train_step",
                        "epoch": epoch,
                        "step": global_step,
                        "loss": loss_value,
                    }
                )

                if global_step % self.config.training.log_every_steps == 0:
                    self.logger.info(
                        "Epoch %s | Step %s | Train loss %.4f",
                        epoch,
                        global_step,
                        loss_value,
                    )

                if global_step % self.config.training.eval_every_steps == 0:
                    eval_loss = self._evaluate(model, eval_loader, device)
                    self.eval_jsonl.write(
                        {
                            "timestamp": utc_now_iso(),
                            "event": "eval_step",
                            "epoch": epoch,
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
                    self.config.training.max_train_steps is not None
                    and global_step >= self.config.training.max_train_steps
                ):
                    stop_training = True
                    break

            epoch_train_loss = sum(step_losses) / len(step_losses) if step_losses else None
            epoch_eval_loss = self._evaluate(model, eval_loader, device)

            if epoch_eval_loss is not None and (
                best_eval_loss is None or epoch_eval_loss < best_eval_loss
            ):
                best_eval_loss = epoch_eval_loss

            epoch_payload = {
                "timestamp": utc_now_iso(),
                "event": "epoch_end",
                "epoch": epoch,
                "step": global_step,
                "train_loss": epoch_train_loss,
                "eval_loss": epoch_eval_loss,
            }
            self.eval_jsonl.write(epoch_payload)

            self.logger.info(
                "Epoch %s complete | train_loss=%s | eval_loss=%s",
                epoch,
                f"{epoch_train_loss:.4f}" if epoch_train_loss is not None else "n/a",
                f"{epoch_eval_loss:.4f}" if epoch_eval_loss is not None else "n/a",
            )

            epoch_results.append(
                {
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "eval_loss": epoch_eval_loss,
                }
            )

            if stop_training:
                break

        total_params = sum(p.numel() for p in model.parameters())

        return {
            "timestamp": utc_now_iso(),
            "device": str(device),
            "global_steps": global_step,
            "epochs_completed": len(epoch_results),
            "best_eval_loss": best_eval_loss,
            "epoch_metrics": epoch_results,
            "num_parameters": total_params,
            "training_seconds": round(time.time() - training_started, 2),
            "train_sequences": len(train_dataset),
            "validation_sequences": len(eval_dataset),
        }

    def _evaluate(
        self,
        model: CausalTransformerLM,
        eval_loader: DataLoader,
        device: torch.device,
    ) -> float | None:
        model.eval()
        losses: list[float] = []

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]
                if loss is None:
                    continue
                losses.append(float(loss.detach().cpu().item()))

        model.train()

        if not losses:
            return None
        return sum(losses) / len(losses)

    def _write_results(self, results: dict[str, Any]) -> None:
        payload = dict(results)
        payload["run_dir"] = str(self.run_dir)
        payload["artifacts_dir"] = str(self.artifacts_dir)
        payload["logs_dir"] = str(self.logs_dir)

        results_path = self.artifacts_dir / "training_results.json"
        results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.logger.info("Saved training results to: %s", results_path)


def main(config_path: str, prepare_only: bool = False) -> dict[str, Any]:
    stage = TrainStage(config_path=config_path, prepare_only=prepare_only)
    return stage.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PikoGPT train stage")
    parser.add_argument("--config", required=True, help="Path to .toml/.yaml config")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run tokenization/splitting/log setup only, skip optimizer training",
    )
    args = parser.parse_args()

    main(config_path=args.config, prepare_only=args.prepare_only)
