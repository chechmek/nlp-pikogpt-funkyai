"""
Build a mixed SFT dataset from ARC-Challenge, CommonsenseQA, and filtered Alpaca.

Output schema is JSONL with Alpaca-compatible fields:
  {"instruction": "...", "input": "", "output": "...", "source": "..."}
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _format_mc_instruction(question: str, labels: list[str], texts: list[str]) -> str:
    lines = [f"Question: {question.strip()}"]
    for label, text in zip(labels, texts, strict=False):
        lines.append(f"{label}) {text.strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


def _normalize_arc_row(row: dict[str, Any]) -> dict[str, str] | None:
    labels = list(row["choices"]["label"])
    texts = list(row["choices"]["text"])
    answer = str(row["answerKey"]).strip().upper()
    if answer not in labels:
        return None
    return {
        "instruction": _format_mc_instruction(str(row["question"]), labels, texts),
        "input": "",
        "output": answer,
        "source": "arc_challenge",
    }


def _normalize_commonsense_row(row: dict[str, Any]) -> dict[str, str] | None:
    labels = list(row["choices"]["label"])
    texts = list(row["choices"]["text"])
    answer = str(row["answerKey"]).strip().upper()
    if answer not in labels:
        return None
    return {
        "instruction": _format_mc_instruction(str(row["question"]), labels, texts),
        "input": "",
        "output": answer,
        "source": "commonsense_qa",
    }


def _normalize_piqa_row(row: dict[str, Any]) -> dict[str, str]:
    labels = ["A", "B"]
    texts = [str(row["sol1"]), str(row["sol2"])]
    answer = labels[int(row["label"])]
    return {
        "instruction": _format_mc_instruction(str(row["goal"]), labels, texts),
        "input": "",
        "output": answer,
        "source": "piqa",
    }


def _normalize_swag_row(row: dict[str, Any]) -> dict[str, str]:
    labels = ["A", "B", "C", "D"]
    endings = [str(row["ending0"]), str(row["ending1"]), str(row["ending2"]), str(row["ending3"])]
    question = f"{str(row['sent1']).strip()} {str(row['sent2']).strip()}".strip()
    answer = labels[int(row["label"])]
    return {
        "instruction": _format_mc_instruction(question, labels, endings),
        "input": "",
        "output": answer,
        "source": "swag",
    }


def _looks_concise_alpaca_output(text: str, max_words: int) -> bool:
    lowered = text.lower()
    if not text.strip():
        return False
    if len(re.findall(r"\S+", text)) > max_words:
        return False
    banned_phrases = [
        "as an ai",
        "language model",
        "i'm sorry",
        "i cannot",
        "i can't",
        "sorry,",
    ]
    if any(phrase in lowered for phrase in banned_phrases):
        return False
    if lowered.count("\n") > 3:
        return False
    return True


def _normalize_alpaca_row(row: dict[str, Any], max_words: int) -> dict[str, str] | None:
    instruction = str(row["instruction"]).strip()
    extra_input = str(row.get("input", "")).strip()
    output = str(row["output"]).strip()
    if not instruction or not _looks_concise_alpaca_output(output, max_words=max_words):
        return None
    return {
        "instruction": instruction,
        "input": extra_input,
        "output": output,
        "source": "alpaca",
    }


def _sample_records(dataset, count: int, seed: int, normalizer, **kwargs) -> list[dict[str, str]]:
    shuffled = dataset.shuffle(seed=seed)
    rows: list[dict[str, str]] = []
    for row in shuffled:
        record = normalizer(row, **kwargs) if kwargs else normalizer(row)
        if record is None:
            continue
        rows.append(record)
        if len(rows) >= count:
            break
    if len(rows) < count:
        raise ValueError(f"Only collected {len(rows)} rows, but {count} were requested")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed SFT dataset from MCQA + Alpaca")
    parser.add_argument("--output-path", type=str, default="data/sft/mixed_arc_cqa_piqa_swag_alpaca_5000.jsonl")
    parser.add_argument("--arc-samples", type=int, default=1000)
    parser.add_argument("--cqa-samples", type=int, default=1000)
    parser.add_argument("--piqa-samples", type=int, default=1000)
    parser.add_argument("--swag-samples", type=int, default=1500)
    parser.add_argument("--alpaca-samples", type=int, default=500)
    parser.add_argument("--alpaca-max-words", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    arc = load_dataset("allenai/ai2_arc", name="ARC-Challenge", split="train")
    cqa = load_dataset("tau/commonsense_qa", split="train")
    piqa = load_dataset("lighteval/piqa", split="train")
    swag = load_dataset("allenai/swag", split="train")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")

    rows = []
    rows.extend(_sample_records(arc, args.arc_samples, args.seed, _normalize_arc_row))
    rows.extend(_sample_records(cqa, args.cqa_samples, args.seed + 1, _normalize_commonsense_row))
    rows.extend(_sample_records(piqa, args.piqa_samples, args.seed + 2, _normalize_piqa_row))
    rows.extend(_sample_records(swag, args.swag_samples, args.seed + 3, _normalize_swag_row))
    rows.extend(
        _sample_records(
            alpaca,
            args.alpaca_samples,
            args.seed + 4,
            _normalize_alpaca_row,
            max_words=args.alpaca_max_words,
        )
    )

    random.Random(args.seed).shuffle(rows)
    output_path = Path(args.output_path).expanduser().resolve()
    _write_jsonl(output_path, rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["source"]] = counts.get(row["source"], 0) + 1

    print(f"Wrote {len(rows)} rows to {output_path}")
    for source, count in sorted(counts.items()):
        print(f"  {source}: {count}")
    print("Example row:")
    print(json.dumps(rows[0], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
