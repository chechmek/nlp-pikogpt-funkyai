"""
SFT Stage — Supervised Fine-Tuning for PikoGPT
================================================
Loads a pre-trained PikoGPT checkpoint and fine-tunes it on the Stanford
Alpaca instruction-response dataset using masked cross-entropy loss.

Key differences from pre-training (VL07):
  - Data: instruction-response pairs instead of raw text
  - Loss: only on response tokens (instruction tokens masked with label=-100)
  - Template: Alpaca format (### Instruction / ### Response)
  - LR: ~10x smaller than pre-training to avoid catastrophic forgetting

Usage:
    python main.py --stage sft --config configs/sft_default.toml --base-checkpoint runs/<run>/artifacts/model_final.pt
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer

from src.training.stage import (
    CausalTransformerLM,
    JsonlWriter,
    load_checkpoint_payload,
    resolve_device,
    safe_exp,
    set_seed,
    utc_now_iso,
)
from src.training.utils import compute_gradient_norm, get_cosine_schedule_with_warmup


# ---------------------------------------------------------------------------
# Alpaca chat template (slide 32: simplest format, no new special tokens)
# ---------------------------------------------------------------------------

_ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

_ALPACA_WITH_INPUT = (
    "Below is an instruction that describes a task, "
    "paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{extra_input}\n\n"
    "### Response:\n"
)


def _format_prompt(instruction: str, extra_input: str = "") -> str:
    if extra_input and extra_input.strip():
        return _ALPACA_WITH_INPUT.format(instruction=instruction, extra_input=extra_input)
    return _ALPACA_NO_INPUT.format(instruction=instruction)


# ---------------------------------------------------------------------------
# SFT Dataset
# ---------------------------------------------------------------------------

class AlpacaSFTDataset(TorchDataset):
    """
    Tokenizes Alpaca instruction-response pairs and applies loss masking.

    Each sample becomes:
      input_ids = [<prompt tokens>  <response tokens> <EOS>] (padded to max_seq_len)
      labels    = [-100 ... -100    <response tokens> <EOS>  -100 ... -100]
                   ^prompt masked^  ^loss computed here^     ^padding masked^

    Because PyTorch cross_entropy ignores index=-100 by default, the existing
    CausalTransformerLM.forward computes loss only on response tokens.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int,
        max_samples: int | None = None,
        val_split: float = 0.05,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        raw = load_dataset("tatsu-lab/alpaca", split="train")

        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))

        # Deterministic train/val split
        split_result = raw.train_test_split(test_size=val_split, seed=seed)
        data = split_result["train"] if split == "train" else split_result["test"]

        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

        self.samples: list[dict[str, list[int]]] = []
        skipped = 0

        for item in data:
            prompt = _format_prompt(item["instruction"], item.get("input", ""))
            response = item["output"]

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            if eos_id is not None:
                response_ids = response_ids + [eos_id]

            full_ids = prompt_ids + response_ids

            # Truncate to max_seq_len
            if len(full_ids) > max_seq_len:
                full_ids = full_ids[:max_seq_len]

            prompt_len = min(len(prompt_ids), len(full_ids))

            # Skip if the response was entirely truncated away
            if prompt_len >= len(full_ids):
                skipped += 1
                continue

            # Masked labels: -100 for prompt, actual token ids for response
            labels = list(full_ids)
            for i in range(prompt_len):
                labels[i] = -100

            # Pad to max_seq_len so batches have uniform shape
            pad_len = max_seq_len - len(full_ids)
            full_ids = full_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len

            self.samples.append({"input_ids": full_ids, "labels": labels})

        if skipped > 0:
            logging.getLogger(__name__).warning(
                "Skipped %d samples where response was fully truncated (max_seq_len=%d)",
                skipped, max_seq_len,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.samples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# SFT Stage
# ---------------------------------------------------------------------------

class SFTStage:
    """Fine-tune a pre-trained PikoGPT checkpoint on Alpaca SFT data."""

    def __init__(
        self,
        base_checkpoint: str | Path,
        run_name: str | None = None,
        # data
        max_samples: int | None = 5000,
        val_split: float = 0.05,
        # training
        device: str = "auto",
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        max_train_steps: int | None = None,
        warmup_steps: int = 50,
        gradient_clip_norm: float = 1.0,
        log_every_steps: int = 10,
        eval_every_steps: int = 50,
        save_every_steps: int | None = 200,
        # misc
        base_dir: str = "runs_sft",
        seed: int = 42,
    ) -> None:
        self.base_checkpoint = Path(base_checkpoint)
        self.max_samples = max_samples
        self.val_split = val_split
        self.device_name = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.max_train_steps = max_train_steps
        self.warmup_steps = warmup_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.log_every_steps = log_every_steps
        self.eval_every_steps = eval_every_steps
        self.save_every_steps = save_every_steps
        self.base_dir = Path(base_dir)
        self.seed = seed

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"sft_alpaca_{timestamp}"
        self.run_dir = self.base_dir / self.run_name
        self.artifacts_dir = self.run_dir / "artifacts"
        self.logs_dir = self.run_dir / "logs"
        self.checkpoints_dir = self.artifacts_dir / "checkpoints"

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._build_logger()
        self.train_jsonl = JsonlWriter(self.logs_dir / "sft_train_metrics.jsonl")
        self.eval_jsonl = JsonlWriter(self.logs_dir / "sft_eval_metrics.jsonl")

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"sft_stage.{self.run_name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.propagate = False

        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        fh = logging.FileHandler(self.logs_dir / "sft_debug.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        return logger

    def _load_base_model(self) -> tuple[CausalTransformerLM, Any]:
        self.logger.info("Loading base checkpoint: %s", self.base_checkpoint)
        payload = load_checkpoint_payload(self.base_checkpoint)

        model_cfg = payload["model"]
        tokenizer_cfg = payload["tokenizer"]

        model = CausalTransformerLM(
            vocab_size=model_cfg["vocab_size"],
            max_seq_len=model_cfg["max_seq_len"],
            n_embd=model_cfg["n_embd"],
            n_layer=model_cfg["n_layer"],
            n_head=model_cfg["n_head"],
            dropout=model_cfg["dropout"],
            layer_norm_epsilon=model_cfg["layer_norm_epsilon"],
            activation=model_cfg.get("activation", "gelu"),
        )
        model.load_state_dict(payload["state_dict"])

        total = sum(p.numel() for p in model.parameters())
        self.logger.info(
            "Loaded model: %dM params | vocab=%d | seq_len=%d",
            total // 1_000_000,
            model_cfg["vocab_size"],
            model_cfg["max_seq_len"],
        )
        return model, tokenizer_cfg

    def _build_tokenizer(self, tokenizer_cfg: dict) -> Any:
        name = tokenizer_cfg.get("name", "gpt2")
        self.logger.info("Loading tokenizer: %s", name)
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        tok.model_max_length = 10_000_000
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def _build_datasets(self, tokenizer, max_seq_len: int) -> tuple[AlpacaSFTDataset, AlpacaSFTDataset]:
        self.logger.info(
            "Building Alpaca SFT dataset (max_samples=%s, max_seq_len=%d)",
            self.max_samples, max_seq_len,
        )
        train_ds = AlpacaSFTDataset(
            tokenizer, max_seq_len,
            max_samples=self.max_samples,
            val_split=self.val_split,
            split="train",
            seed=self.seed,
        )
        val_ds = AlpacaSFTDataset(
            tokenizer, max_seq_len,
            max_samples=self.max_samples,
            val_split=self.val_split,
            split="val",
            seed=self.seed,
        )
        self.logger.info(
            "SFT dataset -> train: %d samples | val: %d samples",
            len(train_ds), len(val_ds),
        )
        return train_ds, val_ds

    @torch.no_grad()
    def _evaluate(self, model: CausalTransformerLM, loader: DataLoader, device: torch.device) -> float | None:
        model.eval()
        loss_sum = 0.0
        count = 0
        use_amp = device.type == "cuda"
        dtype = torch.float16 if use_amp else torch.float32

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                out = model(input_ids=input_ids, labels=labels)
            if out["loss"] is not None:
                loss_sum += float(out["loss"].item())
                count += 1

        model.train()
        return loss_sum / count if count > 0 else None

    def run(self) -> dict[str, Any]:
        started = time.time()
        set_seed(self.seed)

        model, tokenizer_cfg = self._load_base_model()
        tokenizer = self._build_tokenizer(tokenizer_cfg)
        max_seq_len = tokenizer_cfg.get("context_length", model.max_seq_len)

        train_ds, val_ds = self._build_datasets(tokenizer, max_seq_len)

        device = resolve_device(self.device_name)
        model.to(device)

        use_amp = device.type == "cuda"
        amp_dtype = torch.float16 if use_amp else torch.float32
        scaler = torch.amp.GradScaler(enabled=use_amp)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size * 2, shuffle=False)

        if self.max_train_steps is not None:
            total_steps = self.max_train_steps
        else:
            total_steps = math.ceil(len(train_loader)) * self.num_epochs

        optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=0.1,
        )

        self.logger.info("=" * 60)
        self.logger.info(" SFT TRAINING ")
        self.logger.info("=" * 60)
        self.logger.info("  Device:       %s", device)
        self.logger.info("  Samples:      %d train | %d val", len(train_ds), len(val_ds))
        self.logger.info("  Batch size:   %d", self.batch_size)
        self.logger.info("  Total steps:  %d", total_steps)
        self.logger.info("  Learning rate: %.2e  (pre-train used ~3e-4)", self.learning_rate)
        self.logger.info("=" * 60)

        global_step = 0
        best_val_loss: float | None = None
        stop = False

        for epoch in range(1, self.num_epochs + 1):
            if stop:
                break

            model.train()
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(input_ids=input_ids, labels=labels)
                    loss = out["loss"]

                if loss is None:
                    continue

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                grad_norm = compute_gradient_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                current_lr = scheduler.get_last_lr()[0]
                loss_val = float(loss.item())

                if global_step % self.log_every_steps == 0:
                    self.logger.info(
                        "Epoch %d | Step %d | loss %.4f | ppl %.1f | lr %.2e | grad %.4f",
                        epoch, global_step, loss_val, safe_exp(loss_val), current_lr, grad_norm,
                    )
                    self.train_jsonl.write({
                        "timestamp": utc_now_iso(),
                        "epoch": epoch,
                        "step": global_step,
                        "loss": loss_val,
                        "perplexity": safe_exp(loss_val),
                        "lr": current_lr,
                        "grad_norm": grad_norm,
                    })

                if global_step % self.eval_every_steps == 0:
                    val_loss = self._evaluate(model, val_loader, device)
                    val_ppl = safe_exp(val_loss) if val_loss is not None else None
                    self.logger.info(
                        "  [eval] step %d | val_loss %.4f | val_ppl %.1f",
                        global_step,
                        val_loss if val_loss is not None else float("nan"),
                        val_ppl if val_ppl is not None else float("nan"),
                    )
                    self.eval_jsonl.write({
                        "timestamp": utc_now_iso(),
                        "step": global_step,
                        "val_loss": val_loss,
                        "val_perplexity": val_ppl,
                    })
                    if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
                        best_val_loss = val_loss

                if (
                    self.save_every_steps is not None
                    and global_step % self.save_every_steps == 0
                    and global_step < total_steps
                ):
                    self._save_checkpoint(model, tokenizer, tokenizer_cfg, f"step_{global_step:06d}.pt")

                if global_step >= total_steps:
                    stop = True
                    break

        final_path = self._save_checkpoint(model, tokenizer, tokenizer_cfg, "model_final_sft.pt")
        results = {
            "status": "completed",
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "checkpoint_path": str(final_path),
            "duration_seconds": round(time.time() - started, 2),
            "base_checkpoint": str(self.base_checkpoint),
        }
        (self.artifacts_dir / "sft_results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        self.logger.info("SFT complete. Checkpoint: %s", final_path)
        return results

    def _save_checkpoint(
        self,
        model: CausalTransformerLM,
        tokenizer,
        tokenizer_cfg: dict,
        filename: str,
    ) -> Path:
        path = self.artifacts_dir / filename
        torch.save({
            "format": "pikogpt_checkpoint_v2",
            "created_at": utc_now_iso(),
            "checkpoint_kind": "sft",
            "model": {
                "vocab_size": model.vocab_size,
                "max_seq_len": model.max_seq_len,
                "n_embd": model.token_embedding.embedding_dim,
                "n_layer": len(model.transformer.layers),
                "n_head": model.transformer.layers[0].self_attn.num_heads,
                "dropout": model.dropout.p,
                "layer_norm_epsilon": model.final_norm.eps,
                "activation": model.activation,
            },
            "tokenizer": tokenizer_cfg,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        }, path)
        self.logger.info("Saved checkpoint: %s", path)
        return path


def main(
    base_checkpoint: str,
    run_name: str | None = None,
    max_samples: int | None = 5000,
    val_split: float = 0.05,
    device: str = "auto",
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    num_epochs: int = 3,
    max_train_steps: int | None = None,
    warmup_steps: int = 50,
    gradient_clip_norm: float = 1.0,
    log_every_steps: int = 10,
    eval_every_steps: int = 50,
    save_every_steps: int | None = 200,
    base_dir: str = "runs_sft",
    seed: int = 42,
) -> dict[str, Any]:
    stage = SFTStage(
        base_checkpoint=base_checkpoint,
        run_name=run_name,
        max_samples=max_samples,
        val_split=val_split,
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        max_train_steps=max_train_steps,
        warmup_steps=warmup_steps,
        gradient_clip_norm=gradient_clip_norm,
        log_every_steps=log_every_steps,
        eval_every_steps=eval_every_steps,
        save_every_steps=save_every_steps,
        base_dir=base_dir,
        seed=seed,
    )
    return stage.run()
