from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any

import gradio as gr
import torch

from src.inference.stage import _build_model, _generate, _load_checkpoint_payload, _load_tokenizer
from src.training.stage import resolve_device, set_seed


MAX_HISTORY_TURNS = 4


LOGGER = logging.getLogger("pikogpt.chat")


def _discover_checkpoints(checkpoint_dir: str | Path) -> list[str]:
    root = Path(checkpoint_dir).expanduser().resolve()
    if not root.exists():
        return []

    checkpoints = sorted(root.glob("*/artifacts/model_final.pt"))
    return [str(path) for path in checkpoints]


_ALPACA_HEADER = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
)


def _format_prompt(history: list[dict[str, str]], user_message: str) -> str:
    parts: list[str] = [_ALPACA_HEADER]
    for turn in history[-MAX_HISTORY_TURNS:]:
        reply = turn["assistant"].strip()
        if not reply or reply == "[empty response]" or not reply[-1] in ".!?":
            continue
        parts.append(f"### Instruction:\n{turn['user'].strip()}\n\n")
        parts.append(f"### Response:\n{reply}\n\n")
    parts.append(f"### Instruction:\n{user_message.strip()}\n\n### Response:\n")
    return "".join(parts)


def _to_chatbot_messages(history: list[dict[str, str]]) -> list[list[str]]:
    return [[turn["user"], turn["assistant"]] for turn in history]


class ChatSession:
    def __init__(self, device_name: str, seed: int, logger: logging.Logger) -> None:
        self.device_name = device_name
        self.seed = seed
        self.device = resolve_device(device_name)
        self.logger = logger
        self.checkpoint_path: str | None = None
        self.model: torch.nn.Module | None = None
        self.tokenizer = None

    def load_checkpoint(self, checkpoint_path: str) -> str:
        path = str(Path(checkpoint_path).expanduser().resolve())
        if self.checkpoint_path == path and self.model is not None and self.tokenizer is not None:
            return f"Loaded checkpoint: {path}"

        payload = _load_checkpoint_payload(Path(path))
        tokenizer_name = payload["tokenizer"].get("name")
        if not tokenizer_name:
            raise ValueError("Checkpoint tokenizer metadata must include 'name'")

        tokenizer = _load_tokenizer(tokenizer_name=tokenizer_name, quiet=False)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = _build_model(payload["model"])
        model.load_state_dict(payload["state_dict"])
        model.to(self.device)
        model.eval()

        self.checkpoint_path = path
        self.model = model
        self.tokenizer = tokenizer

        return f"Loaded checkpoint: {path}"

    def generate_reply(
        self,
        checkpoint_path: str,
        history: list[dict[str, str]],
        user_message: str,
        max_tokens: int,
        temperature: float,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> tuple[list[dict[str, str]], str]:
        if not user_message.strip():
            return history, "Enter a message."

        status = self.load_checkpoint(checkpoint_path)

        prompt = _format_prompt(history, user_message)
        self.logger.info("Prompt sent to model:\n%s", prompt)
        outputs = _generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            device=self.device,
            top_k=int(top_k),
            top_p=float(top_p),
        )
        reply = outputs["generated_text"].strip()
        last_punct = max(reply.rfind("."), reply.rfind("!"), reply.rfind("?"))
        if last_punct != -1:
            reply = reply[: last_punct + 1]
        if not reply:
            reply = "[empty response]"

        updated_history = history + [{"user": user_message, "assistant": reply}]
        return updated_history, status


def _build_demo(
    checkpoint_dir: str | Path,
    checkpoint_path: str | Path | None,
    device: str,
    max_tokens: int,
    temperature: float,
    seed: int,
):
    discovered = _discover_checkpoints(checkpoint_dir)
    initial_checkpoint = str(checkpoint_path) if checkpoint_path else (discovered[0] if discovered else None)
    if initial_checkpoint is None:
        raise ValueError(
            "No checkpoints found. Pass --checkpoint or place model_final.pt files under --checkpoint-dir."
        )

    choices = [initial_checkpoint] + [path for path in discovered if path != initial_checkpoint]
    session = ChatSession(device_name=device, seed=seed, logger=LOGGER)

    with gr.Blocks(title="PikoGPT Chat") as demo:
        gr.Markdown("# PikoGPT Chat")
        gr.Markdown(
            "Browser-based demo for comparing checkpoints. "
            "The conversation context is preserved until you reset it."
        )

        history_state = gr.State([])

        with gr.Row():
            checkpoint_dropdown = gr.Dropdown(
                choices=choices,
                value=initial_checkpoint,
                label="Checkpoint",
                allow_custom_value=True,
            )
            device_box = gr.Textbox(value=device, label="Device", interactive=False)

        chatbot = gr.Chatbot(label="Conversation", height=500)
        user_input = gr.Textbox(label="Your message", placeholder="Type a prompt...")

        with gr.Row():
            max_tokens_slider = gr.Slider(
                minimum=1,
                maximum=512,
                value=max_tokens,
                step=1,
                label="Max tokens",
            )
            temperature_slider = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                value=temperature,
                step=0.05,
                label="Temperature",
            )
        with gr.Row():
            top_k_slider = gr.Slider(
                minimum=0,
                maximum=200,
                value=0,
                step=1,
                label="Top-K (0 = disabled)",
            )
            top_p_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.05,
                label="Top-P (1.0 = disabled)",
            )

        with gr.Row():
            send_button = gr.Button("Send", variant="primary")
            reset_button = gr.Button("New conversation")

        status_box = gr.Markdown()

        def on_send(
            checkpoint_value: str,
            history: list[dict[str, str]],
            message: str,
            max_tokens_value: int,
            temperature_value: float,
            top_k_value: int,
            top_p_value: float,
        ):
            updated_history, status = session.generate_reply(
                checkpoint_path=checkpoint_value,
                history=history,
                user_message=message,
                max_tokens=int(max_tokens_value),
                temperature=float(temperature_value),
                top_k=int(top_k_value),
                top_p=float(top_p_value),
            )
            return (
                _to_chatbot_messages(updated_history),
                updated_history,
                "",
                status,
            )

        def on_reset():
            return [], [], "Conversation cleared."

        def on_checkpoint_change(checkpoint_value: str):
            try:
                status = session.load_checkpoint(checkpoint_value)
            except Exception as exc:  # pragma: no cover - UI path
                status = f"Failed to load checkpoint: {html.escape(str(exc))}"
            return [], [], status

        send_button.click(
            on_send,
            inputs=[
                checkpoint_dropdown,
                history_state,
                user_input,
                max_tokens_slider,
                temperature_slider,
                top_k_slider,
                top_p_slider,
            ],
            outputs=[chatbot, history_state, user_input, status_box],
        )
        user_input.submit(
            on_send,
            inputs=[
                checkpoint_dropdown,
                history_state,
                user_input,
                max_tokens_slider,
                temperature_slider,
                top_k_slider,
                top_p_slider,
            ],
            outputs=[chatbot, history_state, user_input, status_box],
        )
        reset_button.click(on_reset, outputs=[chatbot, history_state, status_box])
        checkpoint_dropdown.change(
            on_checkpoint_change,
            inputs=[checkpoint_dropdown],
            outputs=[chatbot, history_state, status_box],
        )

    return demo


def main(
    checkpoint_path: str | Path | None = None,
    checkpoint_dir: str | Path = "runs",
    device: str = "auto",
    max_tokens: int = 100,
    temperature: float = 0.8,
    seed: int = 42,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
) -> None:
    log_path = Path("runs/chat_debug.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not LOGGER.handlers:
        LOGGER.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        LOGGER.addHandler(stream_handler)

    demo = _build_demo(
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
    LOGGER.info("Launching chat UI on http://%s:%s", server_name, server_port)
    demo.launch(server_name=server_name, server_port=server_port, share=False)
