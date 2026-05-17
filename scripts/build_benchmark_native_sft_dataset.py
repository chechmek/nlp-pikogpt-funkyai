"""
Build a benchmark-native SFT dataset from the benchmark families themselves.

The output uses raw prompts so SFT can train on the exact evaluation prompt
shapes without the Alpaca wrapper:
  {"prompt": "...", "output": "...", "source": "..."}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _format_mc_prompt(stem: str, labels: list[str], texts: list[str], prefix: str) -> str:
    lines = [f"{prefix}: {stem.strip()}"]
    for label, text in zip(labels, texts, strict=False):
        lines.append(f"{label}) {text.strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


def _format_output(letter: str, answer_text: str, answer_format: str) -> str:
    if answer_format == "label":
        return letter
    if answer_format == "text":
        return answer_text.strip()
    if answer_format == "label_text":
        return f"{letter}) {answer_text.strip()}"
    raise ValueError(f"Unsupported answer_format: {answer_format}")


def _normalize_hellaswag_row(row: dict[str, Any], answer_format: str) -> dict[str, str] | None:
    try:
        label_index = int(row["label"])
    except (TypeError, ValueError):
        return None
    labels = ["A", "B", "C", "D"]
    endings = [str(x).strip() for x in row["endings"]]
    if label_index < 0 or label_index >= len(endings):
        return None
    return {
        "prompt": _format_mc_prompt(str(row["ctx"]), labels, endings, prefix="Context"),
        "output": _format_output(labels[label_index], endings[label_index], answer_format),
        "source": "hellaswag",
    }


def _normalize_openbookqa_row(row: dict[str, Any], answer_format: str) -> dict[str, str] | None:
    labels = list(row["choices"]["label"])
    texts = list(row["choices"]["text"])
    answer = str(row["answerKey"]).strip().upper()
    if answer not in labels:
        return None
    answer_text = texts[labels.index(answer)]
    return {
        "prompt": _format_mc_prompt(str(row["question_stem"]), labels, texts, prefix="Question"),
        "output": _format_output(answer, str(answer_text), answer_format),
        "source": "openbookqa",
    }


def _normalize_winogrande_row(row: dict[str, Any], answer_format: str) -> dict[str, str] | None:
    labels = ["A", "B"]
    texts = [str(row["option1"]), str(row["option2"])]
    answer_idx = int(str(row["answer"]).strip()) - 1
    if answer_idx < 0 or answer_idx >= len(texts):
        return None
    return {
        "prompt": _format_mc_prompt(str(row["sentence"]), labels, texts, prefix="Context"),
        "output": _format_output(labels[answer_idx], texts[answer_idx], answer_format),
        "source": "winogrande",
    }


def _normalize_lambada_row(row: dict[str, Any]) -> dict[str, str] | None:
    text = str(row["text"]).rstrip()
    prompt, sep, last = text.rpartition(" ")
    if not sep:
        return None
    answer = last.strip(" \t\r\n\"'“”‘’.,;:!?()[]{}")
    if not prompt or not answer:
        return None
    return {
        "prompt": prompt,
        "output": answer,
        "source": "lambada",
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
    parser = argparse.ArgumentParser(description="Build a benchmark-native SFT dataset")
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/sft/benchmark_native_5000.jsonl",
    )
    parser.add_argument("--hellaswag-samples", type=int, default=1500)
    parser.add_argument("--openbookqa-samples", type=int, default=1500)
    parser.add_argument("--winogrande-samples", type=int, default=1000)
    parser.add_argument("--lambada-samples", type=int, default=1000)
    parser.add_argument(
        "--mc-answer-format",
        type=str,
        choices=["label", "text", "label_text"],
        default="label_text",
        help="How MC targets are written in the SFT set",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hellaswag = load_dataset("allenai/hellaswag", split="train")
    openbookqa = load_dataset("allenai/openbookqa", name="main", split="train")
    winogrande = load_dataset("allenai/winogrande", name="winogrande_xl", split="train")
    lambada = load_dataset("EleutherAI/lambada_openai", split="test")

    rows: list[dict[str, str]] = []
    rows.extend(
        _sample_records(
            hellaswag,
            args.hellaswag_samples,
            args.seed,
            _normalize_hellaswag_row,
            answer_format=args.mc_answer_format,
        )
    )
    rows.extend(
        _sample_records(
            openbookqa,
            args.openbookqa_samples,
            args.seed + 1,
            _normalize_openbookqa_row,
            answer_format=args.mc_answer_format,
        )
    )
    rows.extend(
        _sample_records(
            winogrande,
            args.winogrande_samples,
            args.seed + 2,
            _normalize_winogrande_row,
            answer_format=args.mc_answer_format,
        )
    )
    rows.extend(_sample_records(lambada, args.lambada_samples, args.seed + 3, _normalize_lambada_row))

    random.Random(args.seed).shuffle(rows)
    output_path = Path(args.output_path).expanduser().resolve()
    _write_jsonl(output_path, rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["source"]] = counts.get(row["source"], 0) + 1

    print(f"Wrote {len(rows)} rows to {output_path}")
    for source, count in sorted(counts.items()):
        print(f"  {source}: {count}")
    print("Example rows:")
    seen = set()
    for row in rows:
        source = row["source"]
        if source in seen:
            continue
        print(json.dumps(row, ensure_ascii=True, indent=2)[:1200])
        seen.add(source)
        if len(seen) >= 4:
            break


if __name__ == "__main__":
    main()
