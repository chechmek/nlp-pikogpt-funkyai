from __future__ import annotations

import argparse
import contextlib
import io
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from src.training.stage import CausalTransformerLM, resolve_device, set_seed


def _load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # `weights_only` exists in newer torch versions only.
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dict")

    state_dict = payload.get("state_dict") or payload.get("model_state_dict")
    if state_dict is None:
        raise ValueError(
            "Checkpoint is missing model weights. Expected key 'state_dict' "
            "or 'model_state_dict'."
        )

    if "model" not in payload or not isinstance(payload["model"], dict):
        raise ValueError("Checkpoint is missing model metadata under key 'model'")
    if "tokenizer" not in payload or not isinstance(payload["tokenizer"], dict):
        raise ValueError("Checkpoint is missing tokenizer metadata under key 'tokenizer'")

    return {
        "state_dict": state_dict,
        "model": payload["model"],
        "tokenizer": payload["tokenizer"],
    }


def _build_model(model_cfg: dict[str, Any]) -> CausalTransformerLM:
    required = [
        "vocab_size",
        "max_seq_len",
        "n_embd",
        "n_layer",
        "n_head",
        "dropout",
        "layer_norm_epsilon",
    ]
    missing = [key for key in required if key not in model_cfg]
    if missing:
        raise ValueError(f"Checkpoint model metadata missing keys: {missing}")

    return CausalTransformerLM(
        vocab_size=int(model_cfg["vocab_size"]),
        max_seq_len=int(model_cfg["max_seq_len"]),
        n_embd=int(model_cfg["n_embd"]),
        n_layer=int(model_cfg["n_layer"]),
        n_head=int(model_cfg["n_head"]),
        dropout=float(model_cfg["dropout"]),
        layer_norm_epsilon=float(model_cfg["layer_norm_epsilon"]),
    )


def _load_tokenizer(tokenizer_name: str, quiet: bool = False):
    if not quiet:
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Leaderboard mode must emit only generated text.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    with contextlib.redirect_stderr(io.StringIO()):
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def _generate(
    model: CausalTransformerLM,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    device: torch.device,
) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)

    if input_ids.shape[1] == 0:
        if tokenizer.eos_token_id is None:
            raise ValueError("Prompt tokenized to empty input and tokenizer has no eos_token_id")
        input_ids = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long, device=device)

    generated_ids: list[int] = []
    model.eval()

    with torch.no_grad():
        for _ in range(max_tokens):
            model_input = input_ids[:, -model.max_seq_len :]
            outputs = model(input_ids=model_input)
            next_token_logits = outputs["logits"][:, -1, :]

            if temperature == 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = next_token_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_ids.append(int(next_token.item()))

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = prompt + generated_text
    return {
        "generated_text": generated_text,
        "full_text": full_text,
        "generated_token_ids": generated_ids,
    }


def main(
    checkpoint_path: str | Path,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    device: str = "auto",
    leaderboard: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    if max_tokens < 0:
        raise ValueError("max_tokens must be >= 0")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")

    set_seed(seed)
    resolved_device = resolve_device(device)

    payload = _load_checkpoint_payload(Path(checkpoint_path))
    tokenizer_name = payload["tokenizer"].get("name")
    if not tokenizer_name:
        raise ValueError("Checkpoint tokenizer metadata must include 'name'")

    tokenizer = _load_tokenizer(tokenizer_name=tokenizer_name, quiet=leaderboard)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _build_model(payload["model"])
    model.load_state_dict(payload["state_dict"])
    model.to(resolved_device)

    result = _generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        device=resolved_device,
    )
    result["device"] = str(resolved_device)

    if leaderboard:
        print(result["generated_text"])
    else:
        print(f"Device: {resolved_device}")
        print("Prompt:")
        print(prompt)
        print("\nGenerated text:")
        print(result["generated_text"])

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PikoGPT inference stage")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (0 = greedy decoding)",
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
        help="Output only generated continuation text",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling",
    )
    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
        leaderboard=args.leaderboard,
        seed=args.seed,
    )
