"""
PikoGPT - Main Entry Point
===========================
Train a small language model from scratch.

Usage:
    python main.py --stage preprocess --num-samples 100000
    python main.py --stage train --config configs/train_default.toml
    python main.py --stage train --config configs/train_default.yaml --prepare-only
    python main.py --stage inference --checkpoint CKPT.pt --prompt "Question: ... Answer:" --max-tokens 1 --temperature 0 --device auto --leaderboard --seed 0
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="PikoGPT - Build an LLM from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required argument: which stage to run
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["preprocess", "train", "inference"],
        help="Which stage to run: preprocess, train, or inference",
    )

    # Preprocessing arguments
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100000,
        help="Number of samples for preprocessing (default: 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for preprocess and inference reproducibility (default: 42)",
    )
    parser.add_argument(
        "--strict-test-data",
        action="store_true",
        help=(
            "Preprocess only: fail if --test-data-path is missing. "
            "By default, preprocess continues without leakage filtering."
        ),
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        default="data/raw/NLP26_OWT_eval/test",
        help="Path to test dataset for filtering",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/openwebtext_clean",
        help="Output path for processed data",
    )
    parser.add_argument(
        "--source-dataset-path",
        type=str,
        default=None,
        help="Preprocess only: local Hugging Face dataset path for input documents",
    )

    # Training arguments
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_default.toml",
        help="Path to training config (.toml/.yaml). Used by --stage train.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Train stage only: run tokenization/splitting/log setup, skip optimizer steps.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Train stage only: resume from a checkpoint path",
    )

    # Inference arguments (for later)
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (for inference)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Input prompt (for inference)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cuda, mps, or cpu",
    )
    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Leaderboard mode: output only generated text",
    )

    args = parser.parse_args()

    # Run the appropriate stage
    if args.stage == "preprocess":
        run_preprocessing(args)
    elif args.stage == "train":
        run_training(args)
    elif args.stage == "inference":
        run_inference(args)


def run_preprocessing(args):
    """Run the preprocessing stage."""
    from src.data.preprocessing import main as preprocess_main

    warning_rule = "ignore:resource_tracker:UserWarning"
    existing_rules = os.environ.get("PYTHONWARNINGS", "")
    if warning_rule not in existing_rules:
        if existing_rules:
            os.environ["PYTHONWARNINGS"] = f"{existing_rules},{warning_rule}"
        else:
            os.environ["PYTHONWARNINGS"] = warning_rule

    preprocess_main(
        num_samples=args.num_samples,
        seed=args.seed,
        test_data_path=args.test_data_path,
        output_path=args.output_path,
        strict_test_data=args.strict_test_data,
        source_dataset_path=args.source_dataset_path,
    )

    # Work around occasional pyarrow teardown hangs on process exit.
    if __name__ == "__main__":
        try:
            from multiprocessing import resource_tracker

            resource_tracker._resource_tracker._stop()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def run_training(args):
    """Run the training stage."""
    from src.training.stage import main as train_main

    train_main(
        config_path=args.config,
        prepare_only=args.prepare_only,
        resume_from_checkpoint=args.resume_from,
    )


def run_inference(args):
    """Run the inference stage."""
    from src.inference.stage import main as inference_main

    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --stage inference")
    if args.prompt is None:
        raise ValueError("--prompt is required for --stage inference")
    if args.max_tokens < 0:
        raise ValueError("--max-tokens must be >= 0")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")

    inference_main(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
        leaderboard=args.leaderboard,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
