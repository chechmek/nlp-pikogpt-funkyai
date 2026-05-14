"""
PikoGPT - Main Entry Point
===========================
Train a small language model from scratch.

Usage:
    python main.py --stage preprocess --num-samples 100000
    python main.py --stage train --config configs/train_default.toml
    python main.py --stage train --config configs/train_default.yaml --prepare-only
    python main.py --stage inference --checkpoint CKPT.pt --prompt "Question: ... Answer:" --max-tokens 1 --temperature 0 --device auto --leaderboard --seed 0
    python main.py --stage chat --checkpoint runs/<run>/artifacts/model_final.pt
    python main.py --stage dpo --base-checkpoint runs/models/model_final_sft.pt --dpo-data-path data/dpo/ultrafeedback_5k.jsonl
    python main.py --stage evaluate --checkpoint CKPT.pt --device auto
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
        choices=["preprocess", "train", "sft", "dpo", "inference", "chat", "evaluate"],
        help="Which stage to run: preprocess, train, sft, dpo, inference, chat, or evaluate",
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

    # SFT arguments
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default=None,
        help="SFT stage: path to pre-trained base model checkpoint",
    )
    parser.add_argument(
        "--sft-max-samples",
        type=int,
        default=5000,
        help="SFT stage: max Alpaca samples to use (default: 5000)",
    )
    parser.add_argument(
        "--sft-lr",
        type=float,
        default=1e-4,
        help="SFT stage: learning rate (default: 1e-4, ~3x lower than pretraining)",
    )
    parser.add_argument(
        "--sft-epochs",
        type=int,
        default=3,
        help="SFT stage: number of epochs (default: 3)",
    )
    parser.add_argument(
        "--sft-max-steps",
        type=int,
        default=1000,
        help="SFT stage: max training steps cap (default: 1000)",
    )
    parser.add_argument(
        "--sft-batch-size",
        type=int,
        default=4,
        help="SFT stage: batch size (default: 4)",
    )
    parser.add_argument(
        "--sft-data-path",
        type=str,
        default=None,
        help="SFT stage: optional local JSONL dataset in Alpaca-style instruction/input/output schema",
    )
    parser.add_argument(
        "--dpo-data-path",
        type=str,
        default=None,
        help="DPO stage: path to local JSONL preference dataset with prompt/chosen/rejected",
    )
    parser.add_argument(
        "--dpo-max-samples",
        type=int,
        default=None,
        help="DPO stage: optional cap on loaded preference pairs",
    )
    parser.add_argument(
        "--dpo-beta",
        type=float,
        default=0.1,
        help="DPO stage: beta coefficient for preference sharpness (default: 0.1)",
    )
    parser.add_argument(
        "--dpo-lr",
        type=float,
        default=5e-6,
        help="DPO stage: learning rate (default: 5e-6)",
    )
    parser.add_argument(
        "--dpo-epochs",
        type=int,
        default=1,
        help="DPO stage: number of epochs (default: 1)",
    )
    parser.add_argument(
        "--dpo-max-steps",
        type=int,
        default=200,
        help="DPO stage: max training steps cap (default: 200)",
    )
    parser.add_argument(
        "--dpo-batch-size",
        type=int,
        default=2,
        help="DPO stage: batch size (default: 2)",
    )
    parser.add_argument(
        "--dpo-val-split",
        type=float,
        default=0.05,
        help="DPO stage: validation split fraction (default: 0.05)",
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
        default=32,
        help="Maximum tokens to generate (default: 32)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
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
    parser.add_argument(
        "--debug-next-token",
        action="store_true",
        help="Inference stage only: print first-step next-token probabilities before generating",
    )
    parser.add_argument(
        "--debug-top-n",
        type=int,
        default=20,
        help="Inference stage only: how many next-token candidates to print in debug mode",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="runs",
        help="Chat stage only: root directory scanned for model checkpoints",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Chat stage only: host for the local web UI",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Chat stage only: port for the local web UI",
    )
    parser.add_argument(
        "--owt-test-path",
        type=str,
        default="src/data/raw/NLP26_OWT_eval/test",
        help="Evaluate stage only: path to official OpenWebText test split",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="Evaluate stage only: evaluation batch size",
    )
    parser.add_argument(
        "--owt-max-docs",
        type=int,
        default=None,
        help="Evaluate stage only: optional cap on OWT test documents for quicker comparisons",
    )
    parser.add_argument(
        "--rebuild-eval-cache",
        action="store_true",
        help="Evaluate stage only: rebuild cached packed eval datasets",
    )

    args = parser.parse_args()

    # Run the appropriate stage
    if args.stage == "preprocess":
        run_preprocessing(args)
    elif args.stage == "train":
        run_training(args)
    elif args.stage == "sft":
        run_sft(args)
    elif args.stage == "dpo":
        run_dpo(args)
    elif args.stage == "inference":
        run_inference(args)
    elif args.stage == "chat":
        run_chat(args)
    elif args.stage == "evaluate":
        run_evaluate(args)


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


def run_sft(args):
    """Run the SFT stage."""
    from src.sft.stage import main as sft_main

    if not args.base_checkpoint:
        raise ValueError("--base-checkpoint is required for --stage sft")

    sft_main(
        base_checkpoint=args.base_checkpoint,
        data_path=args.sft_data_path,
        max_samples=args.sft_max_samples,
        device=args.device,
        batch_size=args.sft_batch_size,
        learning_rate=args.sft_lr,
        num_epochs=args.sft_epochs,
        max_train_steps=args.sft_max_steps,
        seed=args.seed,
    )


def run_training(args):
    """Run the training stage."""
    from src.training.stage import main as train_main

    train_main(
        config_path=args.config,
        prepare_only=args.prepare_only,
        resume_from_checkpoint=args.resume_from,
    )


def run_dpo(args):
    """Run the DPO stage."""
    from src.dpo.stage import main as dpo_main

    if not args.base_checkpoint:
        raise ValueError("--base-checkpoint is required for --stage dpo")
    if not args.dpo_data_path:
        raise ValueError("--dpo-data-path is required for --stage dpo")

    dpo_main(
        base_checkpoint=args.base_checkpoint,
        data_path=args.dpo_data_path,
        max_samples=args.dpo_max_samples,
        val_split=args.dpo_val_split,
        device=args.device,
        batch_size=args.dpo_batch_size,
        learning_rate=args.dpo_lr,
        beta=args.dpo_beta,
        num_epochs=args.dpo_epochs,
        max_train_steps=args.dpo_max_steps,
        seed=args.seed,
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
        debug_next_token=args.debug_next_token,
        debug_top_n=args.debug_top_n,
    )


def run_chat(args):
    """Run the chat demo stage."""
    from src.chat.stage import main as chat_main

    chat_main(
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        server_name=args.server_name,
        server_port=args.server_port,
    )


def run_evaluate(args):
    """Run standalone checkpoint evaluation."""
    from src.evaluation.stage import main as evaluate_main

    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --stage evaluate")

    evaluate_main(
        checkpoint_path=args.checkpoint,
        device=args.device,
        seed=args.seed,
        owt_test_path=args.owt_test_path,
        eval_batch_size=args.eval_batch_size,
        owt_max_docs=args.owt_max_docs,
        rebuild_eval_cache=args.rebuild_eval_cache,
    )


if __name__ == "__main__":
    main()
