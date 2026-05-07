"""
Prepare local DPO JSONL subsets from HuggingFaceH4/ultrafeedback_binarized.

Writes compact preference-pair files with schema:
  {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _extract_last_assistant_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for item in reversed(messages):
        if isinstance(item, dict) and item.get("role") == "assistant":
            content = item.get("content")
            if isinstance(content, str):
                return content.strip()
    return ""


def _extract_prompt(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()

    messages = row.get("messages")
    if isinstance(messages, list):
        for item in reversed(messages):
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return ""


def _build_record(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = _extract_prompt(row)
    chosen = _extract_last_assistant_text(row.get("chosen"))
    rejected = _extract_last_assistant_text(row.get("rejected"))
    if not prompt or not chosen or not rejected or chosen == rejected:
        return None

    record = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }
    if row.get("prompt_id") is not None:
        record["prompt_id"] = row["prompt_id"]
    if row.get("score_chosen") is not None:
        record["score_chosen"] = row["score_chosen"]
    if row.get("score_rejected") is not None:
        record["score_rejected"] = row["score_rejected"]
    return record


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local UltraFeedback DPO subsets")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dpo",
        help="Directory where local JSONL subsets will be written",
    )
    parser.add_argument(
        "--smoke-samples",
        type=int,
        default=500,
        help="Number of preference pairs for the smoke subset",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=5000,
        help="Number of preference pairs for the main subset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed used before slicing subsets",
    )
    args = parser.parse_args()

    max_needed = max(args.smoke_samples, args.train_samples)
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    dataset = dataset.shuffle(seed=args.seed)

    rows: list[dict[str, Any]] = []
    for row in dataset:
        record = _build_record(row)
        if record is None:
            continue
        rows.append(record)
        if len(rows) >= max_needed:
            break

    if len(rows) < max_needed:
        raise ValueError(
            f"Only built {len(rows)} usable preference pairs, but {max_needed} were requested"
        )

    output_dir = Path(args.output_dir)
    smoke_path = output_dir / f"ultrafeedback_{args.smoke_samples}.jsonl"
    train_path = output_dir / f"ultrafeedback_{args.train_samples}.jsonl"
    _write_jsonl(smoke_path, rows[:args.smoke_samples])
    _write_jsonl(train_path, rows[:args.train_samples])

    print(f"Wrote {args.smoke_samples} pairs to {smoke_path}")
    print(f"Wrote {args.train_samples} pairs to {train_path}")


if __name__ == "__main__":
    main()
