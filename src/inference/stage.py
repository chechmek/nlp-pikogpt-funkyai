from __future__ import annotations

import argparse
import contextlib
import io
import logging
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from src.training.stage import CausalTransformerLM, resolve_device, set_seed


EOS_LOGIT_DIVISOR = 1.5
UPPERCASE_LETTER_LOGIT_DIVISOR = 0
TOKEN_LOGIT_DIVISORS: dict[int, float] = {
    32: 0.5,
    33: UPPERCASE_LETTER_LOGIT_DIVISOR,  # B
    34: UPPERCASE_LETTER_LOGIT_DIVISOR,  # C
    35: UPPERCASE_LETTER_LOGIT_DIVISOR,  # D
    36: UPPERCASE_LETTER_LOGIT_DIVISOR,  # E
    37: UPPERCASE_LETTER_LOGIT_DIVISOR,  # F
    38: UPPERCASE_LETTER_LOGIT_DIVISOR,  # G
    39: UPPERCASE_LETTER_LOGIT_DIVISOR,  # H
    40: UPPERCASE_LETTER_LOGIT_DIVISOR,  # I
    41: UPPERCASE_LETTER_LOGIT_DIVISOR,  # J
    42: UPPERCASE_LETTER_LOGIT_DIVISOR,  # K
    43: UPPERCASE_LETTER_LOGIT_DIVISOR,  # L
    44: UPPERCASE_LETTER_LOGIT_DIVISOR,  # M
    45: UPPERCASE_LETTER_LOGIT_DIVISOR,  # N
    46: UPPERCASE_LETTER_LOGIT_DIVISOR,  # O
    47: UPPERCASE_LETTER_LOGIT_DIVISOR,  # P
    48: UPPERCASE_LETTER_LOGIT_DIVISOR,  # Q
    49: UPPERCASE_LETTER_LOGIT_DIVISOR,  # R
    50: UPPERCASE_LETTER_LOGIT_DIVISOR,  # S
    51: UPPERCASE_LETTER_LOGIT_DIVISOR,  # T
    52: UPPERCASE_LETTER_LOGIT_DIVISOR,  # U
    53: UPPERCASE_LETTER_LOGIT_DIVISOR,  # V
    54: UPPERCASE_LETTER_LOGIT_DIVISOR,  # W
    55: UPPERCASE_LETTER_LOGIT_DIVISOR,  # X
    56: UPPERCASE_LETTER_LOGIT_DIVISOR,  # Y
    57: UPPERCASE_LETTER_LOGIT_DIVISOR,  # Z
    447: 1,
}
REPETITION_PENALTY = 1.3
NO_REPEAT_NGRAM_SIZE = 4


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
        activation=str(model_cfg.get("activation", "gelu")),
    )


def _load_tokenizer(tokenizer_name: str, quiet: bool = False):
    if not quiet:
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Leaderboard mode must emit only generated text.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    with contextlib.redirect_stderr(io.StringIO()):
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, top_k).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Shift right so the token that pushes over the threshold is kept
    sorted_indices_to_remove = (cumulative_probs - torch.softmax(sorted_logits, dim=-1)) >= top_p
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return logits.scatter(-1, sorted_indices, sorted_logits)


_STOP_STRINGS = []


def _get_stop_sequences(tokenizer) -> list[list[int]]:
    seqs = []
    for s in _STOP_STRINGS:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            seqs.append(ids)
    return seqs


def _matches_stop(generated_ids: list[int], stop_seqs: list[list[int]]) -> list[int] | None:
    for seq in stop_seqs:
        if len(generated_ids) >= len(seq) and generated_ids[-len(seq):] == seq:
            return seq
    return None


def _generate(
    model: CausalTransformerLM,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    device: torch.device,
    top_k: int = 0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)

    if input_ids.shape[1] == 0:
        if tokenizer.eos_token_id is None:
            raise ValueError("Prompt tokenized to empty input and tokenizer has no eos_token_id")
        input_ids = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long, device=device)

    stop_seqs = _get_stop_sequences(tokenizer)
    generated_ids: list[int] = []
    finish_reason = "max_tokens"
    stop_sequence_text = None
    model.eval()

    with torch.no_grad():
        for _ in range(max_tokens):
            model_input = input_ids[:, -model.max_seq_len :]
            outputs = model(input_ids=model_input)
            next_token_logits = outputs["logits"][:, -1, :]
            next_token_logits = _apply_token_penalties(next_token_logits, tokenizer, generated_ids)

            if temperature == 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = next_token_logits / temperature
                scaled_logits = _apply_top_k(scaled_logits, top_k)
                scaled_logits = _apply_top_p(scaled_logits, top_p)
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
                finish_reason = "eos"
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_ids.append(int(next_token.item()))

            matched = _matches_stop(generated_ids, stop_seqs)
            if matched:
                generated_ids = generated_ids[:-len(matched)]
                finish_reason = "stop_sequence"
                stop_sequence_text = tokenizer.decode(matched, skip_special_tokens=False)
                break

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = prompt + generated_text
    return {
        "generated_text": generated_text,
        "full_text": full_text,
        "generated_token_ids": generated_ids,
        "finish_reason": finish_reason,
        "stop_sequence_text": stop_sequence_text,
    }


def _format_token_debug_text(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if not text:
        return "<empty>"
    return text


def _apply_repetition_penalty(logits: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    if REPETITION_PENALTY <= 1.0 or not token_ids:
        return logits

    adjusted = logits.clone()
    for token_id in set(token_ids):
        value = adjusted[:, token_id]
        adjusted[:, token_id] = torch.where(
            value < 0,
            value * REPETITION_PENALTY,
            value / REPETITION_PENALTY,
        )
    return adjusted


def _get_banned_ngram_tokens(token_ids: list[int]) -> set[int]:
    if NO_REPEAT_NGRAM_SIZE <= 0 or len(token_ids) + 1 < NO_REPEAT_NGRAM_SIZE:
        return set()
    if NO_REPEAT_NGRAM_SIZE == 1:
        return set(token_ids)

    prefix = token_ids[-(NO_REPEAT_NGRAM_SIZE - 1) :]
    banned: set[int] = set()
    window = NO_REPEAT_NGRAM_SIZE - 1
    for idx in range(len(token_ids) - window):
        if token_ids[idx : idx + window] == prefix:
            banned.add(token_ids[idx + window])
    return banned


def _apply_token_penalties(logits: torch.Tensor, tokenizer, generated_ids: list[int]) -> torch.Tensor:
    adjusted = logits.clone()
    if tokenizer.eos_token_id is not None and EOS_LOGIT_DIVISOR > 1.0:
        adjusted[:, tokenizer.eos_token_id] = adjusted[:, tokenizer.eos_token_id] / EOS_LOGIT_DIVISOR
    for token_id, divisor in TOKEN_LOGIT_DIVISORS.items():
        if divisor > 1.0 and 0 <= token_id < adjusted.shape[-1]:
            adjusted[:, token_id] = adjusted[:, token_id] / divisor
    adjusted = _apply_repetition_penalty(adjusted, generated_ids)
    banned_tokens = _get_banned_ngram_tokens(generated_ids)
    for token_id in banned_tokens:
        if 0 <= token_id < adjusted.shape[-1]:
            adjusted[:, token_id] = float("-inf")
    return adjusted


def _debug_next_token_distribution(
    model: CausalTransformerLM,
    tokenizer,
    prompt: str,
    temperature: float,
    device: torch.device,
    top_n: int = 20,
) -> None:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)

    if input_ids.shape[1] == 0:
        if tokenizer.eos_token_id is None:
            raise ValueError("Prompt tokenized to empty input and tokenizer has no eos_token_id")
        input_ids = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        model_input = input_ids[:, -model.max_seq_len :]
        logits = model(input_ids=model_input)["logits"][:, -1, :][0]
    logits = logits.clone()
    if tokenizer.eos_token_id is not None and EOS_LOGIT_DIVISOR > 1.0:
        logits[tokenizer.eos_token_id] = logits[tokenizer.eos_token_id] / EOS_LOGIT_DIVISOR
    for token_id, divisor in TOKEN_LOGIT_DIVISORS.items():
        if divisor > 1.0 and 0 <= token_id < logits.shape[-1]:
            logits[token_id] = logits[token_id] / divisor

    sampling_logits = logits if temperature == 0 else logits / temperature
    probs = torch.softmax(sampling_logits, dim=-1)
    top_n = max(1, min(int(top_n), probs.shape[-1]))
    top_probs, top_indices = torch.topk(probs, k=top_n, dim=-1)

    eos_token_id = tokenizer.eos_token_id
    eos_prob = float(probs[eos_token_id].item()) if eos_token_id is not None else None
    eos_rank = None
    if eos_token_id is not None:
        eos_rank = int((probs > probs[eos_token_id]).sum().item()) + 1

    print(f"Device: {device}")
    print("Prompt:")
    print(prompt)
    print("\nNext-token debug:")
    print(f"Temperature: {temperature}")
    print(f"EOS logit divisor: {EOS_LOGIT_DIVISOR}")
    if TOKEN_LOGIT_DIVISORS:
        print(f"Other token divisors: {TOKEN_LOGIT_DIVISORS}")
    print(f"Repetition penalty: {REPETITION_PENALTY}")
    if eos_token_id is not None:
        eos_text = _format_token_debug_text(tokenizer, eos_token_id)
        print(f"EOS token: id={eos_token_id} text={eos_text!r} rank={eos_rank} prob={eos_prob:.6f}")
    else:
        print("EOS token: tokenizer has no eos_token_id")

    print(f"\nTop {top_n} next-token candidates:")
    for rank, (token_id, prob) in enumerate(
        zip(top_indices.tolist(), top_probs.tolist(), strict=False),
        start=1,
    ):
        token_text = _format_token_debug_text(tokenizer, int(token_id))
        logit = float(logits[int(token_id)].item())
        marker = " <EOS>" if eos_token_id is not None and int(token_id) == eos_token_id else ""
        print(
            f"{rank:>2}. id={int(token_id):>6} prob={float(prob):.6f} "
            f"logit={logit:.4f} token={token_text!r}{marker}"
        )


def _parse_mc_options(prompt: str) -> list[tuple[str, str]]:
    matches = re.findall(r"(?m)^([A-E])\)\s*(.+)$", prompt)
    return [(letter, text.strip()) for letter, text in matches]


def _score_continuation(
    model: CausalTransformerLM,
    tokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if not continuation_ids:
        return float("-inf")

    full_ids = prompt_ids + continuation_ids
    if len(full_ids) < 2:
        return float("-inf")

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    model_input = input_ids[:, -model.max_seq_len :]
    with torch.no_grad():
        logits = model(input_ids=model_input)["logits"]

    truncated_ids = model_input[0].tolist()
    truncated_prompt_len = min(len(prompt_ids), len(truncated_ids))
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = model_input[:, 1:]

    score = 0.0
    for pos in range(target_ids.shape[1]):
        target_index = pos + 1
        if target_index < truncated_prompt_len:
            continue
        token_id = target_ids[0, pos].item()
        score += float(log_probs[0, pos, token_id].item())
    return score


def _predict_mc_letter(
    model: CausalTransformerLM,
    tokenizer,
    prompt: str,
    device: torch.device,
) -> str | None:
    options = _parse_mc_options(prompt)
    if len(options) < 2:
        return None

    best_letter: str | None = None
    best_score = float("-inf")
    for letter, text in options:
        continuation = f" {letter}) {text}"
        score = _score_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuation=continuation,
            device=device,
        )
        if score > best_score:
            best_score = score
            best_letter = letter
    return best_letter


def main(
    checkpoint_path: str | Path,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    device: str = "auto",
    leaderboard: bool = False,
    seed: int = 42,
    debug_next_token: bool = False,
    debug_top_n: int = 20,
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
    model.eval()

    if debug_next_token:
        _debug_next_token_distribution(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            temperature=temperature,
            device=resolved_device,
            top_n=debug_top_n,
        )
        print()

    if leaderboard:
        mc_letter = _predict_mc_letter(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=resolved_device,
        )
        if mc_letter is not None:
            result = {
                "generated_text": mc_letter,
                "full_text": prompt + mc_letter,
                "generated_token_ids": [],
                "device": str(resolved_device),
            }
            print(result["generated_text"])
            return result

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
    parser.add_argument(
        "--debug-next-token",
        action="store_true",
        help="Print the first-step next-token distribution before generating",
    )
    parser.add_argument(
        "--debug-top-n",
        type=int,
        default=20,
        help="How many next-token candidates to print in debug mode",
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
        debug_next_token=args.debug_next_token,
        debug_top_n=args.debug_top_n,
    )
