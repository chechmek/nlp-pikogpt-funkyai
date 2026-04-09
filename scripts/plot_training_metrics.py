from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return pd.DataFrame(rows)


def load_jsonl_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return load_jsonl(path)


def save_loss_plot(train_df: pd.DataFrame, eval_df: pd.DataFrame, output_path: Path, title: str) -> None:
    if train_df.empty and eval_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    if not train_df.empty and {"step", "loss"}.issubset(train_df.columns):
        ax.plot(train_df["step"], train_df["loss"], label="train loss", linewidth=2)
        plotted = True

    if not eval_df.empty and {"step", "eval_loss"}.issubset(eval_df.columns):
        eval_steps = eval_df[eval_df["event"] == "eval_step"].copy()
        if not eval_steps.empty:
            ax.plot(eval_steps["step"], eval_steps["eval_loss"], label="eval loss", linewidth=2)
            plotted = True

        epoch_end = eval_df[eval_df["event"] == "epoch_end"].copy()
        if not epoch_end.empty:
            ax.scatter(epoch_end["step"], epoch_end["eval_loss"], label="epoch end", s=30)
            plotted = True

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_lr_plot(train_df: pd.DataFrame, output_path: Path, title: str) -> None:
    if train_df.empty or not {"step", "lr"}.issubset(train_df.columns):
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_df["step"], train_df["lr"], linewidth=2)
    ax.set_title(f"{title} - Learning Rate")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_grad_plot(train_df: pd.DataFrame, output_path: Path, title: str) -> None:
    if train_df.empty or not {"step", "grad_norm"}.issubset(train_df.columns):
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_df["step"], train_df["grad_norm"], linewidth=2)
    ax.set_title(f"{title} - Gradient Norm")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient norm")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_summary(run_dir: Path, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> dict[str, object]:
    training_results_path = run_dir / "artifacts" / "training_results.json"
    training_results = {}
    if training_results_path.exists():
        training_results = json.loads(training_results_path.read_text(encoding="utf-8"))

    summary: dict[str, object] = {
        "run_dir": str(run_dir),
        "num_train_points": int(len(train_df)),
        "num_eval_points": int(len(eval_df)),
        "last_train_loss": None,
        "best_eval_loss": None,
        "final_eval_loss": None,
    }

    if not train_df.empty and "loss" in train_df.columns:
        summary["last_train_loss"] = float(train_df["loss"].iloc[-1])

    if not eval_df.empty and "eval_loss" in eval_df.columns:
        eval_values = eval_df["eval_loss"].dropna()
        if not eval_values.empty:
            summary["best_eval_loss"] = float(eval_values.min())
            summary["final_eval_loss"] = float(eval_values.iloc[-1])

    for key in [
        "status",
        "device",
        "global_step",
        "global_steps",
        "epochs_completed",
        "num_parameters",
        "world_size",
        "grad_accum_steps",
        "training_seconds",
        "checkpoint_path",
    ]:
        if key in training_results:
            summary[key] = training_results[key]

    return summary


def main(run_dir: str, output_dir: str | None = None, title: str | None = None) -> None:
    run_path = Path(run_dir).resolve()
    logs_dir = run_path / "logs"
    out_dir = Path(output_dir).resolve() if output_dir else run_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_jsonl_optional(logs_dir / "train_metrics.jsonl")
    eval_df = load_jsonl_optional(logs_dir / "eval_metrics.jsonl")
    chart_title = title or run_path.name

    save_loss_plot(train_df, eval_df, out_dir / "loss_curve.png", chart_title)
    save_lr_plot(train_df, out_dir / "learning_rate_curve.png", chart_title)
    save_grad_plot(train_df, out_dir / "gradient_norm_curve.png", chart_title)

    summary = build_summary(run_path, train_df, eval_df)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if train_df.empty and eval_df.empty:
        print(f"No metric logs found for: {run_path}")
        print(f"Saved summary to: {out_dir / 'summary.json'}")
        return

    print(f"Saved plots to: {out_dir}")
    print(f"Saved summary to: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot train/eval metrics for a PikoGPT run")
    parser.add_argument("--run-dir", required=True, help="Run directory containing logs/")
    parser.add_argument("--output-dir", default=None, help="Optional destination directory")
    parser.add_argument("--title", default=None, help="Optional plot title")
    args = parser.parse_args()

    main(run_dir=args.run_dir, output_dir=args.output_dir, title=args.title)
