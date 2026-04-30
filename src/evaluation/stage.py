from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader, TensorDataset

from src.inference.stage import _build_model, _load_checkpoint_payload, _load_tokenizer
from src.training.stage import resolve_device, safe_exp, set_seed


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"pikogpt.evaluate.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_dir / "evaluate.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _evaluate_packed_sequences(
    model: torch.nn.Module,
    sequences: list[list[int]],
    eval_batch_size: int,
    device: torch.device,
) -> float | None:
    if not sequences:
        return None

    input_ids_tensor = torch.tensor(sequences, dtype=torch.long)
    labels_tensor = input_ids_tensor.clone()
    dataset = TensorDataset(input_ids_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)

    model.eval()
    loss_sum = 0.0
    batch_count = 0.0
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else torch.float32

    with torch.no_grad():
        for batch_input_ids, batch_labels in loader:
            batch_input_ids = batch_input_ids.to(device)
            batch_labels = batch_labels.to(device)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=batch_input_ids, labels=batch_labels)
            loss = outputs["loss"]
            if loss is not None:
                loss_sum += float(loss.detach().item())
                batch_count += 1.0

    model.train()
    if batch_count == 0:
        return None
    return loss_sum / batch_count


def _pack_text_dataset(
    rows: list[str],
    tokenizer,
    block_size: int,
    append_eos_token: bool,
) -> list[list[int]]:
    eos_id = tokenizer.eos_token_id
    all_tokens: list[int] = []

    for text in rows:
        if not text or not text.strip():
            continue
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_tokens.extend(encoded)
        if append_eos_token and eos_id is not None:
            all_tokens.append(eos_id)

    usable = (len(all_tokens) // block_size) * block_size
    if usable == 0:
        return []

    return [
        all_tokens[i : i + block_size]
        for i in range(0, usable, block_size)
    ]


def _eval_cache_path(
    cache_name: str,
    tokenizer_name: str,
    block_size: int,
    append_eos_token: bool,
    doc_limit: int | None = None,
) -> Path:
    safe_tokenizer = tokenizer_name.replace("/", "_")
    suffix = f"_docs{doc_limit}" if doc_limit is not None else "_docsall"
    return Path("runs/eval_cache") / f"{cache_name}_{safe_tokenizer}_ctx{block_size}_eos{int(append_eos_token)}{suffix}"


def _save_packed_sequences(sequences: list[list[int]], cache_path: Path) -> None:
    dataset = Dataset.from_dict(
        {"input_ids": sequences, "labels": [seq.copy() for seq in sequences]}
    )
    if cache_path.exists():
        shutil.rmtree(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_path))


def _load_packed_sequences(cache_path: Path) -> list[list[int]]:
    dataset = load_from_disk(str(cache_path))
    return [row["input_ids"] for row in dataset]


def _evaluate_wikitext(
    model: torch.nn.Module,
    tokenizer,
    block_size: int,
    eval_batch_size: int,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, float | None]:
    logger.info("Running WikiText-103 evaluation...")
    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    except Exception as exc:
        logger.warning("Could not load WikiText-103: %s", exc)
        return {"wikitext_loss": None, "wikitext_perplexity": None}

    texts = [row.get("text", "") for row in dataset]
    sequences = _pack_text_dataset(
        rows=texts,
        tokenizer=tokenizer,
        block_size=block_size,
        append_eos_token=True,
    )
    if not sequences:
        logger.warning("WikiText-103 produced 0 packed sequences")
        return {"wikitext_loss": None, "wikitext_perplexity": None}

    logger.info("WikiText-103: %s sequences of length %s", f"{len(sequences):,}", block_size)
    avg_loss = _evaluate_packed_sequences(model, sequences, eval_batch_size, device)
    if avg_loss is None:
        return {"wikitext_loss": None, "wikitext_perplexity": None}
    ppl = safe_exp(avg_loss)
    logger.info("WikiText-103 | loss=%.4f | perplexity=%.2f", avg_loss, ppl)
    return {"wikitext_loss": avg_loss, "wikitext_perplexity": ppl}


def _evaluate_owt_test(
    model: torch.nn.Module,
    tokenizer,
    block_size: int,
    eval_batch_size: int,
    device: torch.device,
    test_data_path: str | Path,
    append_eos_token: bool,
    logger: logging.Logger,
    tokenizer_name: str,
    max_docs: int | None = None,
    rebuild_cache: bool = False,
) -> dict[str, float | None]:
    path = Path(test_data_path).expanduser().resolve()
    logger.info("Running OpenWebText test evaluation from: %s", path)
    if not path.exists():
        logger.warning("OpenWebText test path not found: %s", path)
        return {"owt_test_loss": None, "owt_test_perplexity": None}

    cache_path = _eval_cache_path(
        cache_name="owt_test",
        tokenizer_name=tokenizer_name,
        block_size=block_size,
        append_eos_token=append_eos_token,
        doc_limit=max_docs,
    )
    if cache_path.exists() and not rebuild_cache:
        logger.info("Loading cached packed OpenWebText eval set from: %s", cache_path)
        sequences = _load_packed_sequences(cache_path)
        logger.info("Loaded cached OpenWebText eval set: %s sequences", f"{len(sequences):,}")
        avg_loss = _evaluate_packed_sequences(model, sequences, eval_batch_size, device)
        if avg_loss is None:
            return {"owt_test_loss": None, "owt_test_perplexity": None}
        ppl = safe_exp(avg_loss)
        logger.info("OpenWebText test | loss=%.4f | perplexity=%.2f", avg_loss, ppl)
        return {"owt_test_loss": avg_loss, "owt_test_perplexity": ppl}

    logger.info("Loading OpenWebText test dataset from disk...")
    dataset = load_from_disk(str(path))
    if max_docs is not None:
        limit = min(len(dataset), max_docs)
        dataset = dataset.select(range(limit))
        logger.info("Capped OpenWebText test dataset to %s documents", f"{len(dataset):,}")
    else:
        logger.info("Loaded OpenWebText test dataset with %s documents", f"{len(dataset):,}")

    if rebuild_cache and cache_path.exists():
        logger.info("Rebuilding cached OpenWebText eval set at: %s", cache_path)

    logger.info("Tokenizing and packing OpenWebText test documents...")
    eos_id = tokenizer.eos_token_id
    all_tokens: list[int] = []
    processed_docs = 0

    for doc in dataset:
        text = doc.get("text", "")
        if text and text.strip():
            encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_tokens.extend(encoded)
            if append_eos_token and eos_id is not None:
                all_tokens.append(eos_id)
        processed_docs += 1
        if processed_docs % 1000 == 0:
            logger.info(
                "OpenWebText tokenization progress: %s docs | %s tokens accumulated",
                f"{processed_docs:,}",
                f"{len(all_tokens):,}",
            )

    logger.info("Finished tokenization: %s docs | %s tokens total", f"{processed_docs:,}", f"{len(all_tokens):,}")
    usable = (len(all_tokens) // block_size) * block_size
    logger.info("Packing tokens into fixed-length sequences of %s...", block_size)
    sequences = [
        all_tokens[i : i + block_size]
        for i in range(0, usable, block_size)
    ]
    if not sequences:
        logger.warning("OpenWebText test produced 0 packed sequences")
        return {"owt_test_loss": None, "owt_test_perplexity": None}

    logger.info("OpenWebText test: %s sequences of length %s", f"{len(sequences):,}", block_size)
    logger.info("Saving cached packed OpenWebText eval set to: %s", cache_path)
    _save_packed_sequences(sequences, cache_path)
    avg_loss = _evaluate_packed_sequences(model, sequences, eval_batch_size, device)
    if avg_loss is None:
        return {"owt_test_loss": None, "owt_test_perplexity": None}
    ppl = safe_exp(avg_loss)
    logger.info("OpenWebText test | loss=%.4f | perplexity=%.2f", avg_loss, ppl)
    return {"owt_test_loss": avg_loss, "owt_test_perplexity": ppl}


def _checkpoint_label(checkpoint_path: Path) -> str:
    if checkpoint_path.parent.name == "artifacts":
        return checkpoint_path.parent.parent.name
    return checkpoint_path.stem


def main(
    checkpoint_path: str | Path,
    device: str = "auto",
    seed: int = 42,
    owt_test_path: str | Path = "src/data/raw/NLP26_OWT_eval/test",
    eval_batch_size: int = 64,
    owt_max_docs: int | None = None,
    rebuild_eval_cache: bool = False,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    label = _checkpoint_label(checkpoint_path)
    run_dir = Path("runs/evals") / label
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(run_dir)

    set_seed(seed)
    resolved_device = resolve_device(device)
    logger.info("Loading checkpoint: %s", checkpoint_path)
    logger.info("Using device: %s", resolved_device)

    payload = _load_checkpoint_payload(checkpoint_path)
    tokenizer_name = payload["tokenizer"].get("name")
    if not tokenizer_name:
        raise ValueError("Checkpoint tokenizer metadata must include 'name'")

    tokenizer = _load_tokenizer(tokenizer_name=tokenizer_name, quiet=False)
    tokenizer.model_max_length = 10_000_000
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _build_model(payload["model"])
    model.load_state_dict(payload["state_dict"])
    model.to(resolved_device)

    block_size = int(payload["tokenizer"].get("context_length", payload["model"]["max_seq_len"]))
    append_eos_token = bool(payload["tokenizer"].get("append_eos_token", True))

    results: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "checkpoint_path": str(checkpoint_path),
        "device": str(resolved_device),
        "eval_batch_size": eval_batch_size,
        "owt_test_path": str(Path(owt_test_path).expanduser()),
        "owt_max_docs": owt_max_docs,
        "model": payload["model"],
        "tokenizer": payload["tokenizer"],
    }

    results.update(
        _evaluate_wikitext(
            model=model,
            tokenizer=tokenizer,
            block_size=block_size,
            eval_batch_size=eval_batch_size,
            device=resolved_device,
            logger=logger,
        )
    )
    results.update(
        _evaluate_owt_test(
            model=model,
            tokenizer=tokenizer,
            block_size=block_size,
            eval_batch_size=eval_batch_size,
            device=resolved_device,
            test_data_path=owt_test_path,
            append_eos_token=append_eos_token,
            logger=logger,
            tokenizer_name=tokenizer_name,
            max_docs=owt_max_docs,
            rebuild_cache=rebuild_eval_cache,
        )
    )

    results_path = run_dir / "evaluation_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved evaluation results to: %s", results_path)

    jsonl = JsonlWriter(run_dir / "evaluation_metrics.jsonl")
    jsonl.write(results)

    print(json.dumps(results, indent=2))
    return results
