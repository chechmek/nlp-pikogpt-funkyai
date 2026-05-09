"""
Build a benchmark-native MC-only SFT dataset with shuffled answer order.

Output rows use exact benchmark-style prompts and letter-only targets:
  {"prompt": "...", "output": "B", "source": "..."}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _format_prompt(prefix: str, stem: str, choices: list[tuple[str, str]]) -> str:
    lines = [f"{prefix}: {stem.strip()}"]
    for label, text in choices:
        lines.append(f"{label}) {text.strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


def _shuffle_choices(
    texts: list[str],
    correct_index: int,
    labels: list[str],
    rng: random.Random,
) -> tuple[list[tuple[str, str]], str]:
    items = list(enumerate(texts))
    rng.shuffle(items)
    shuffled_choices: list[tuple[str, str]] = []
    gold_label = ""
    for new_idx, (old_idx, text) in enumerate(items):
        label = labels[new_idx]
        shuffled_choices.append((label, text))
        if old_idx == correct_index:
            gold_label = label
    if not gold_label:
        raise ValueError("Failed to remap correct label after shuffling")
    return shuffled_choices, gold_label


def _normalize_hellaswag_row(row: dict[str, Any], rng: random.Random) -> dict[str, str] | None:
    try:
        correct_index = int(row["label"])
    except (TypeError, ValueError):
        return None
    endings = [str(x).strip() for x in row["endings"]]
    if correct_index < 0 or correct_index >= len(endings):
        return None
    labels = ["A", "B", "C", "D"]
    shuffled_choices, gold_label = _shuffle_choices(endings, correct_index, labels, rng)
    return {
        "prompt": _format_prompt("Context", str(row["ctx"]), shuffled_choices),
        "output": gold_label,
        "source": "hellaswag",
    }


def _normalize_openbookqa_row(row: dict[str, Any], rng: random.Random) -> dict[str, str] | None:
    labels = ["A", "B", "C", "D"]
    texts = [str(x).strip() for x in row["choices"]["text"]]
    answer = str(row["answerKey"]).strip().upper()
    raw_labels = [str(x).strip().upper() for x in row["choices"]["label"]]
    if answer not in raw_labels:
        return None
    correct_index = raw_labels.index(answer)
    shuffled_choices, gold_label = _shuffle_choices(texts, correct_index, labels, rng)
    return {
        "prompt": _format_prompt("Question", str(row["question_stem"]), shuffled_choices),
        "output": gold_label,
        "source": "openbookqa",
    }


def _normalize_winogrande_row(row: dict[str, Any], rng: random.Random) -> dict[str, str] | None:
    labels = ["A", "B"]
    texts = [str(row["option1"]).strip(), str(row["option2"]).strip()]
    correct_index = int(str(row["answer"]).strip()) - 1
    if correct_index < 0 or correct_index >= len(texts):
        return None
    shuffled_choices, gold_label = _shuffle_choices(texts, correct_index, labels, rng)
    return {
        "prompt": _format_prompt("Context", str(row["sentence"]), shuffled_choices),
        "output": gold_label,
        "source": "winogrande",
    }


def _sample_records(dataset, count: int, seed: int, normalizer) -> list[dict[str, str]]:
    shuffled = dataset.shuffle(seed=seed)
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for row in shuffled:
        record = normalizer(row, rng)
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
    parser = argparse.ArgumentParser(description="Build an MC-only benchmark-native SFT dataset")
    parser.add_argument("--output-path", type=str, default="data/sft/benchmark_mc_only_5000.jsonl")
    parser.add_argument("--hellaswag-samples", type=int, default=2000)
    parser.add_argument("--openbookqa-samples", type=int, default=2000)
    parser.add_argument("--winogrande-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hellaswag = load_dataset("allenai/hellaswag", split="train")
    openbookqa = load_dataset("allenai/openbookqa", name="main", split="train")
    winogrande = load_dataset("allenai/winogrande", name="winogrande_xl", split="train")

    rows: list[dict[str, str]] = []
    rows.extend(_sample_records(hellaswag, args.hellaswag_samples, args.seed, _normalize_hellaswag_row))
    rows.extend(_sample_records(openbookqa, args.openbookqa_samples, args.seed + 1, _normalize_openbookqa_row))
    rows.extend(_sample_records(winogrande, args.winogrande_samples, args.seed + 2, _normalize_winogrande_row))

    random.Random(args.seed).shuffle(rows)
    output_path = Path(args.output_path).expanduser().resolve()
    _write_jsonl(output_path, rows)

    source_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    for row in rows:
        source = row["source"]
        source_counts[source] = source_counts.get(source, 0) + 1
        label = row["output"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"Wrote {len(rows)} rows to {output_path}")
    print("Sources:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")
    print("Labels:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print("Example rows:")
    seen = set()
    for row in rows:
        source = row["source"]
        if source in seen:
            continue
        print(json.dumps(row, ensure_ascii=True, indent=2)[:1200])
        seen.add(source)
        if len(seen) >= 3:
            break


if __name__ == "__main__":
    main()
