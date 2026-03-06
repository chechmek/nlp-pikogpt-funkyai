# Preprocessing Stage

## Purpose

Build a clean OpenWebText dataset for training and reduce data leakage risk by removing sentences that appear in the evaluation test split.

Implementation: `src/data/preprocessing.py`

## What It Does

1. Loads NLP26 test split from disk.
2. Extracts and hashes test sentences.
3. Streams OpenWebText from Hugging Face.
4. Filters non-English documents.
5. Removes test-leakage sentences.
6. Cleans HTML, URLs, code blocks, and noisy characters.
7. Applies quality filters and minimum length checks.
8. Saves cleaned dataset as Hugging Face `Dataset` on disk.

## Run Command

From repository root:

```bash
PYTHONPATH=. .venv/bin/python main.py \
  --stage preprocess \
  --num-samples 100000 \
  --seed 42 \
  --test-data-path src/data/raw/NLP26_OWT_eval/test \
  --output-path src/data/processed/openwebtext_clean
```

## CLI Arguments

- `--num-samples`: number of clean samples to keep.
- `--seed`: random seed for reproducibility.
- `--test-data-path`: path to NLP26 test split used for leakage filtering.
- `--output-path`: destination folder for processed dataset.

## Output

At `--output-path`, Hugging Face dataset artifacts are written, typically:

- `data-00000-of-00001.arrow`
- `dataset_info.json`
- `state.json`

## Notes

- First run can take time because sentence hashes are built for all test documents.
- Internet access is required to stream OpenWebText from Hugging Face.
