"""
PikoGPT - Main Entry Point
===========================
Train a small language model from scratch.

Usage:
    python main.py --stage preprocess --num-samples 100000
    python main.py --stage train --config configs/train_default.toml
    python main.py --stage train --config configs/train_default.yaml --prepare-only
    python main.py --stage inference --checkpoint model.pt --prompt "Hello"
"""

import argparse


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
        help="Random seed for reproducibility (default: 42)",
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

    preprocess_main(
        num_samples=args.num_samples,
        seed=args.seed,
        test_data_path=args.test_data_path,
        output_path=args.output_path,
    )


def run_training(args):
    """Run the training stage."""
    from src.training.stage import main as train_main

    train_main(
        config_path=args.config,
        prepare_only=args.prepare_only,
    )


def run_inference(args):
    """Run the inference stage."""
    print("Inference stage not yet implemented.")
    print("This will be added in future weeks!")


if __name__ == "__main__":
    main()
