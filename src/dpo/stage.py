"""
DPO Stage — Direct Preference Optimization for PikoGPT
======================================================
Loads an SFT-trained PikoGPT checkpoint and aligns it further on local
preference pairs using a frozen reference model and the standard DPO loss.

Expected dataset format: JSONL with fields
  {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer

from src.training.stage import (
    CausalTransformerLM,
    JsonlWriter,
    load_checkpoint_payload,
    resolve_device,
    set_seed,
    utc_now_iso,
)
from src.training.utils import compute_gradient_norm, get_cosine_schedule_with_warmup


SYSTEM_PROMPT = "Question: {question}\nAnswer:"


def _format_prompt(prompt: str) -> str:
    return SYSTEM_PROMPT.format(question=prompt.strip())


class PreferenceJsonlDataset(TorchDataset):
    """Tokenize local preference pairs and mask prompt tokens from scoring."""

    def __init__(
        self,
        records: list[dict[str, str]],
        tokenizer,
        max_seq_len: int,
    ) -> None:
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
        self.samples: list[dict[str, list[int]]] = []
        skipped = 0

        for item in records:
            prompt_text = _format_prompt(item["prompt"])
            chosen = item["chosen"].strip()
            rejected = item["rejected"].strip()
            if not prompt_text.strip() or not chosen or not rejected:
                skipped += 1
                continue

            chosen_sample = self._build_sample(
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                response_text=chosen,
                max_seq_len=max_seq_len,
                pad_id=pad_id,
                eos_id=eos_id,
            )
            rejected_sample = self._build_sample(
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                response_text=rejected,
                max_seq_len=max_seq_len,
                pad_id=pad_id,
                eos_id=eos_id,
            )

            if chosen_sample is None or rejected_sample is None:
                skipped += 1
                continue

            self.samples.append({
                "chosen_input_ids": chosen_sample["input_ids"],
                "chosen_labels": chosen_sample["labels"],
                "rejected_input_ids": rejected_sample["input_ids"],
                "rejected_labels": rejected_sample["labels"],
            })

        if skipped > 0:
            logging.getLogger(__name__).warning(
                "Skipped %d preference pairs due to empty or fully truncated responses",
                skipped,
            )

    @staticmethod
    def _build_sample(
        tokenizer,
        prompt_text: str,
        response_text: str,
        max_seq_len: int,
        pad_id: int | None,
        eos_id: int | None,
    ) -> dict[str, list[int]] | None:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = tokenizer.encode(response_text, add_special_tokens=False)
        if eos_id is not None:
            response_ids = response_ids + [eos_id]

        full_ids = prompt_ids + response_ids
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]

        prompt_len = min(len(prompt_ids), len(full_ids))
        if prompt_len >= len(full_ids):
            return None

        labels = list(full_ids)
        for i in range(prompt_len):
            labels[i] = -100

        pad_len = max_seq_len - len(full_ids)
        full_ids = full_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len
        return {"input_ids": full_ids, "labels": labels}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.samples[idx]
        return {
            "chosen_input_ids": torch.tensor(item["chosen_input_ids"], dtype=torch.long),
            "chosen_labels": torch.tensor(item["chosen_labels"], dtype=torch.long),
            "rejected_input_ids": torch.tensor(item["rejected_input_ids"], dtype=torch.long),
            "rejected_labels": torch.tensor(item["rejected_labels"], dtype=torch.long),
        }


def _load_jsonl_records(path: Path, max_samples: int | None = None) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            prompt = str(payload.get("prompt", "")).strip()
            chosen = str(payload.get("chosen", "")).strip()
            rejected = str(payload.get("rejected", "")).strip()
            if prompt and chosen and rejected:
                records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
            if max_samples is not None and len(records) >= max_samples:
                break
    if not records:
        raise ValueError(f"No valid preference pairs found in {path}")
    return records


def _split_records(
    records: list[dict[str, str]],
    val_split: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < val_split < 0.5:
        raise ValueError("val_split must be between 0 and 0.5")
    if len(records) < 2:
        raise ValueError("Need at least 2 preference pairs for train/validation split")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_split))
    if val_count >= len(shuffled):
        val_count = len(shuffled) - 1
    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]
    return train_records, val_records


def _sequence_logprob_from_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~valid_mask, 0)
    token_logps = F.log_softmax(shift_logits, dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    token_logps = token_logps * valid_mask
    return token_logps.sum(dim=-1)


def _dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    policy_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    advantages = policy_logratios - ref_logratios
    losses = -F.logsigmoid(beta * advantages)

    reward_chosen = beta * (policy_chosen_logp - ref_chosen_logp)
    reward_rejected = beta * (policy_rejected_logp - ref_rejected_logp)
    pref_accuracy = (advantages > 0).float().mean()

    return losses.mean(), {
        "policy_logratio": float(policy_logratios.detach().mean().item()),
        "ref_logratio": float(ref_logratios.detach().mean().item()),
        "advantage": float(advantages.detach().mean().item()),
        "reward_margin": float((reward_chosen - reward_rejected).detach().mean().item()),
        "preference_accuracy": float(pref_accuracy.detach().item()),
    }


class DPOStage:
    def __init__(
        self,
        base_checkpoint: str | Path,
        data_path: str | Path,
        run_name: str | None = None,
        max_samples: int | None = None,
        val_split: float = 0.05,
        device: str = "auto",
        batch_size: int = 2,
        learning_rate: float = 5e-6,
        weight_decay: float = 0.01,
        beta: float = 0.1,
        num_epochs: int = 1,
        max_train_steps: int | None = 200,
        warmup_steps: int = 10,
        gradient_clip_norm: float = 1.0,
        log_every_steps: int = 10,
        eval_every_steps: int = 50,
        save_every_steps: int | None = 200,
        base_dir: str = "runs_dpo",
        seed: int = 42,
    ) -> None:
        self.base_checkpoint = Path(base_checkpoint)
        self.data_path = Path(data_path)
        self.max_samples = max_samples
        self.val_split = val_split
        self.device_name = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta = beta
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
        self.run_name = run_name or f"dpo_{self.data_path.stem}_{timestamp}"
        self.run_dir = self.base_dir / self.run_name
        self.artifacts_dir = self.run_dir / "artifacts"
        self.logs_dir = self.run_dir / "logs"
        self.checkpoints_dir = self.artifacts_dir / "checkpoints"

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._build_logger()
        self.train_jsonl = JsonlWriter(self.logs_dir / "dpo_train_metrics.jsonl")
        self.eval_jsonl = JsonlWriter(self.logs_dir / "dpo_eval_metrics.jsonl")

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"dpo_stage.{self.run_name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.propagate = False

        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        fh = logging.FileHandler(self.logs_dir / "dpo_debug.log", encoding="utf-8")
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
        return model, tokenizer_cfg

    def _build_tokenizer(self, tokenizer_cfg: dict[str, Any]) -> Any:
        name = tokenizer_cfg.get("name", "gpt2")
        self.logger.info("Loading tokenizer: %s", name)
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        tok.model_max_length = 10_000_000
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def _build_datasets(self, tokenizer, max_seq_len: int) -> tuple[PreferenceJsonlDataset, PreferenceJsonlDataset]:
        self.logger.info("Loading local preference pairs from %s", self.data_path)
        records = _load_jsonl_records(self.data_path, max_samples=self.max_samples)
        train_records, val_records = _split_records(records, val_split=self.val_split, seed=self.seed)
        train_ds = PreferenceJsonlDataset(train_records, tokenizer=tokenizer, max_seq_len=max_seq_len)
        val_ds = PreferenceJsonlDataset(val_records, tokenizer=tokenizer, max_seq_len=max_seq_len)
        self.logger.info(
            "DPO dataset -> train: %d pairs | val: %d pairs",
            len(train_ds),
            len(val_ds),
        )
        return train_ds, val_ds

    def _forward_logps(
        self,
        model: CausalTransformerLM,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        use_amp: bool,
        amp_dtype: torch.dtype,
        no_grad: bool = False,
    ) -> torch.Tensor:
        context = torch.no_grad() if no_grad else torch.enable_grad()
        with context:
            with torch.amp.autocast(device_type=input_ids.device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(input_ids=input_ids)["logits"]
            return _sequence_logprob_from_labels(logits=logits, labels=labels)

    @torch.no_grad()
    def _evaluate(
        self,
        policy_model: CausalTransformerLM,
        reference_model: CausalTransformerLM,
        loader: DataLoader,
        device: torch.device,
    ) -> dict[str, float] | None:
        if len(loader) == 0:
            return None

        policy_model.eval()
        reference_model.eval()
        use_amp = device.type == "cuda"
        amp_dtype = torch.float16 if use_amp else torch.float32

        metric_sums = {
            "loss": 0.0,
            "policy_logratio": 0.0,
            "ref_logratio": 0.0,
            "advantage": 0.0,
            "reward_margin": 0.0,
            "preference_accuracy": 0.0,
        }
        count = 0

        for batch in loader:
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            policy_chosen_logp = self._forward_logps(
                model=policy_model,
                input_ids=chosen_input_ids,
                labels=chosen_labels,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                no_grad=True,
            )
            policy_rejected_logp = self._forward_logps(
                model=policy_model,
                input_ids=rejected_input_ids,
                labels=rejected_labels,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                no_grad=True,
            )
            ref_chosen_logp = self._forward_logps(
                model=reference_model,
                input_ids=chosen_input_ids,
                labels=chosen_labels,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                no_grad=True,
            )
            ref_rejected_logp = self._forward_logps(
                model=reference_model,
                input_ids=rejected_input_ids,
                labels=rejected_labels,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                no_grad=True,
            )

            loss, metrics = _dpo_loss(
                policy_chosen_logp=policy_chosen_logp,
                policy_rejected_logp=policy_rejected_logp,
                ref_chosen_logp=ref_chosen_logp,
                ref_rejected_logp=ref_rejected_logp,
                beta=self.beta,
            )
            metric_sums["loss"] += float(loss.item())
            for key in ("policy_logratio", "ref_logratio", "advantage", "reward_margin", "preference_accuracy"):
                metric_sums[key] += metrics[key]
            count += 1

        policy_model.train()
        metrics_out = {key: value / count for key, value in metric_sums.items()}
        return metrics_out

    def run(self) -> dict[str, Any]:
        started = time.time()
        set_seed(self.seed)

        policy_model, tokenizer_cfg = self._load_base_model()
        reference_model = copy.deepcopy(policy_model)
        for param in reference_model.parameters():
            param.requires_grad = False

        tokenizer = self._build_tokenizer(tokenizer_cfg)
        max_seq_len = tokenizer_cfg.get("context_length", policy_model.max_seq_len)
        train_ds, val_ds = self._build_datasets(tokenizer, max_seq_len)

        device = resolve_device(self.device_name)
        policy_model.to(device)
        reference_model.to(device)
        reference_model.eval()

        use_amp = device.type == "cuda"
        amp_dtype = torch.float16 if use_amp else torch.float32
        scaler = torch.amp.GradScaler(enabled=use_amp)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size * 2, shuffle=False)

        if self.max_train_steps is not None:
            total_steps = self.max_train_steps
        else:
            total_steps = math.ceil(len(train_loader)) * self.num_epochs

        optimizer = AdamW(policy_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=0.1,
        )

        baseline_eval = self._evaluate(policy_model, reference_model, val_loader, device)
        self.logger.info("=" * 60)
        self.logger.info(" DPO TRAINING ")
        self.logger.info("=" * 60)
        self.logger.info("  Device:        %s", device)
        self.logger.info("  Pairs:         %d train | %d val", len(train_ds), len(val_ds))
        self.logger.info("  Batch size:    %d", self.batch_size)
        self.logger.info("  Total steps:   %d", total_steps)
        self.logger.info("  Learning rate: %.2e", self.learning_rate)
        self.logger.info("  Beta:          %.3f", self.beta)
        if baseline_eval is not None:
            self.logger.info(
                "  Baseline val:  loss %.4f | pref_acc %.3f | margin %.4f",
                baseline_eval["loss"],
                baseline_eval["preference_accuracy"],
                baseline_eval["reward_margin"],
            )
        self.logger.info("=" * 60)

        global_step = 0
        best_val_loss: float | None = None
        stop = False

        for epoch in range(1, self.num_epochs + 1):
            if stop:
                break

            policy_model.train()
            for batch in train_loader:
                chosen_input_ids = batch["chosen_input_ids"].to(device)
                chosen_labels = batch["chosen_labels"].to(device)
                rejected_input_ids = batch["rejected_input_ids"].to(device)
                rejected_labels = batch["rejected_labels"].to(device)

                policy_chosen_logp = self._forward_logps(
                    model=policy_model,
                    input_ids=chosen_input_ids,
                    labels=chosen_labels,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                policy_rejected_logp = self._forward_logps(
                    model=policy_model,
                    input_ids=rejected_input_ids,
                    labels=rejected_labels,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                ref_chosen_logp = self._forward_logps(
                    model=reference_model,
                    input_ids=chosen_input_ids,
                    labels=chosen_labels,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    no_grad=True,
                )
                ref_rejected_logp = self._forward_logps(
                    model=reference_model,
                    input_ids=rejected_input_ids,
                    labels=rejected_labels,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    no_grad=True,
                )

                loss, metrics = _dpo_loss(
                    policy_chosen_logp=policy_chosen_logp,
                    policy_rejected_logp=policy_rejected_logp,
                    ref_chosen_logp=ref_chosen_logp,
                    ref_rejected_logp=ref_rejected_logp,
                    beta=self.beta,
                )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = compute_gradient_norm(policy_model)
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), self.gradient_clip_norm)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                current_lr = scheduler.get_last_lr()[0]
                loss_val = float(loss.item())

                if global_step % self.log_every_steps == 0:
                    self.logger.info(
                        "Epoch %d | Step %d | loss %.4f | pref_acc %.3f | adv %.4f | lr %.2e | grad %.4f",
                        epoch,
                        global_step,
                        loss_val,
                        metrics["preference_accuracy"],
                        metrics["advantage"],
                        current_lr,
                        grad_norm,
                    )
                    self.train_jsonl.write({
                        "timestamp": utc_now_iso(),
                        "epoch": epoch,
                        "step": global_step,
                        "loss": loss_val,
                        "preference_accuracy": metrics["preference_accuracy"],
                        "advantage": metrics["advantage"],
                        "reward_margin": metrics["reward_margin"],
                        "policy_logratio": metrics["policy_logratio"],
                        "ref_logratio": metrics["ref_logratio"],
                        "lr": current_lr,
                        "grad_norm": grad_norm,
                    })

                if global_step % self.eval_every_steps == 0:
                    val_metrics = self._evaluate(policy_model, reference_model, val_loader, device)
                    if val_metrics is not None:
                        self.logger.info(
                            "  [eval] step %d | val_loss %.4f | val_pref_acc %.3f | val_margin %.4f",
                            global_step,
                            val_metrics["loss"],
                            val_metrics["preference_accuracy"],
                            val_metrics["reward_margin"],
                        )
                        self.eval_jsonl.write({
                            "timestamp": utc_now_iso(),
                            "step": global_step,
                            **val_metrics,
                        })
                        if best_val_loss is None or val_metrics["loss"] < best_val_loss:
                            best_val_loss = val_metrics["loss"]

                if (
                    self.save_every_steps is not None
                    and global_step % self.save_every_steps == 0
                    and global_step < total_steps
                ):
                    self._save_checkpoint(policy_model, tokenizer_cfg, f"step_{global_step:06d}.pt")

                if global_step >= total_steps:
                    stop = True
                    break

        final_val = self._evaluate(policy_model, reference_model, val_loader, device)
        if final_val is not None and (best_val_loss is None or final_val["loss"] < best_val_loss):
            best_val_loss = final_val["loss"]
        final_path = self._save_checkpoint(policy_model, tokenizer_cfg, "model_final_dpo.pt")
        results = {
            "status": "completed",
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "checkpoint_path": str(final_path),
            "duration_seconds": round(time.time() - started, 2),
            "base_checkpoint": str(self.base_checkpoint),
            "data_path": str(self.data_path),
            "beta": self.beta,
            "final_val_metrics": final_val,
            "baseline_val_metrics": baseline_eval,
        }
        (self.artifacts_dir / "dpo_results.json").write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        self.logger.info("DPO complete. Checkpoint: %s", final_path)
        return results

    def _save_checkpoint(
        self,
        model: CausalTransformerLM,
        tokenizer_cfg: dict[str, Any],
        filename: str,
    ) -> Path:
        path = self.artifacts_dir / filename
        torch.save({
            "format": "pikogpt_checkpoint_v2",
            "created_at": utc_now_iso(),
            "checkpoint_kind": "dpo",
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
    data_path: str,
    run_name: str | None = None,
    max_samples: int | None = None,
    val_split: float = 0.05,
    device: str = "auto",
    batch_size: int = 2,
    learning_rate: float = 5e-6,
    weight_decay: float = 0.01,
    beta: float = 0.1,
    num_epochs: int = 1,
    max_train_steps: int | None = 200,
    warmup_steps: int = 10,
    gradient_clip_norm: float = 1.0,
    log_every_steps: int = 10,
    eval_every_steps: int = 50,
    save_every_steps: int | None = 200,
    base_dir: str = "runs_dpo",
    seed: int = 42,
) -> dict[str, Any]:
    stage = DPOStage(
        base_checkpoint=base_checkpoint,
        data_path=data_path,
        run_name=run_name,
        max_samples=max_samples,
        val_split=val_split,
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta=beta,
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
