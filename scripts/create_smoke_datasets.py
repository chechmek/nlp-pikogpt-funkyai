from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datasets import Dataset


OVERLAP_SENTENCE = (
    "The archive room smelled like dust, old glue, and rain that had dried into the concrete walls."
)


def long_doc(topic: str, detail: str) -> str:
    return (
        f"{OVERLAP_SENTENCE} "
        f"{topic} {detail} "
        "Every paragraph is intentionally long and grammatical so the preprocessing filters keep it, "
        "the tokenizer has enough material to pack into multiple blocks, and the smoke run stays fully local."
    )


def write_dataset(path: Path, texts: list[str]) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict({"text": texts}).save_to_disk(str(path))


def main(root: str) -> None:
    root_path = Path(root)
    source_path = root_path / "source_dataset"
    test_path = root_path / "test_dataset"

    source_texts = [
        long_doc("A careful engineer wrote notes about gradient accumulation and minibatch boundaries.", "The notes explained why synchronized updates matter in distributed training."),
        long_doc("A quiet library server streamed plain English articles into a temporary cache.", "The cache was small enough for a smoke test but large enough to validate the full preprocessing path."),
        long_doc("Several students compared checkpoint files after a late evening run.", "They verified that the optimizer state, scheduler state, and model weights all moved together."),
        long_doc("An observatory team practiced inference prompts on a toy language model.", "The prompt was short, the decoding length was tiny, and the expected output only needed to prove the checkpoint loaded."),
        long_doc("A systems notebook described process groups, local ranks, and synchronized samplers.", "It also warned that every rank should not preprocess the same corpus independently."),
        long_doc("The whiteboard included a sketch of compile backends for CUDA, CPU, and fallback paths.", "The team wanted one code path that remained valid on laptops and GPU servers."),
        long_doc("A mock deployment checklist mentioned token packing, validation loss, and reproducible seeds.", "Each item was ordinary, but together they caught the class of errors that waste GPU time."),
        long_doc("The hallway conversation turned into a debugging session about data leakage and sentence hashing.", "Someone noticed that a repeated benchmark sentence had to be removed before training."),
        long_doc("The nightly smoke suite was intentionally plain, local, and deterministic.", "Its job was not to reach good model quality, only to prove that the end to end pipeline stayed executable."),
        long_doc("A teaching assistant wrote a long paragraph about padding tokens and end of sequence markers.", "That paragraph existed mainly so the preprocessing stage had enough surviving text to keep."),
        long_doc("An experiment tracker recorded checkpoints at short intervals for fast resume tests.", "Those files were small, but they forced the pipeline to preserve enough state for continuation."),
        long_doc("The final paragraph talked about inference after training and asked for one short continuation.", "Nothing about the content mattered as much as making the full pipeline observable."),
    ]

    test_texts = [
        OVERLAP_SENTENCE + " This sentence appears in the held out test set to exercise leakage filtering.",
        "A second held out record exists so the test dataset looks realistic and the sentence hash table is not empty.",
    ]

    write_dataset(source_path, source_texts)
    write_dataset(test_path, test_texts)

    print(f"source_dataset={source_path}")
    print(f"test_dataset={test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create local smoke datasets for preprocess/train/inference runs")
    parser.add_argument(
        "--root",
        default="data/smoke",
        help="Root directory where source_dataset and test_dataset will be written",
    )
    args = parser.parse_args()
    main(root=args.root)
