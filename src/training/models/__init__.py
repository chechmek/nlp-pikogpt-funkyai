"""Model architectures for PikoGPT."""

from .causal_transformer import CausalTransformerLM
from .llama import LlamaModel

__all__ = ["CausalTransformerLM", "LlamaModel", "build_model_from_config", "build_model_from_checkpoint"]


def build_model_from_config(config, vocab_size: int, max_seq_len: int):
    """Build model based on config.model.architecture field."""
    arch = getattr(config.model, "architecture", "causal_transformer")

    if arch == "causal_transformer":
        return CausalTransformerLM(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            n_embd=config.model.n_embd,
            n_layer=config.model.n_layer,
            n_head=config.model.n_head,
            dropout=config.model.dropout,
            layer_norm_epsilon=config.model.layer_norm_epsilon,
            activation=config.model.activation,
        )
    elif arch == "llama":
        n_kv_head = getattr(config.model, "n_kv_head", None) or config.model.n_head
        ffn_dim = getattr(config.model, "ffn_dim", None) or int(8 / 3 * config.model.n_embd)
        rope_base = getattr(config.model, "rope_base", 10000.0)
        return LlamaModel(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            n_embd=config.model.n_embd,
            n_layer=config.model.n_layer,
            n_head=config.model.n_head,
            n_kv_head=n_kv_head,
            ffn_dim=ffn_dim,
            dropout=config.model.dropout,
            eps=config.model.layer_norm_epsilon,
            rope_base=rope_base,
        )
    else:
        raise ValueError(
            f"Unknown architecture '{arch}'. Choose 'causal_transformer' or 'llama'."
        )


def build_model_from_checkpoint(model_cfg: dict):
    """Build model from checkpoint metadata dict."""
    arch = model_cfg.get("architecture", "causal_transformer")

    if arch == "causal_transformer":
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
    elif arch == "llama":
        return LlamaModel(
            vocab_size=int(model_cfg["vocab_size"]),
            max_seq_len=int(model_cfg["max_seq_len"]),
            n_embd=int(model_cfg["n_embd"]),
            n_layer=int(model_cfg["n_layer"]),
            n_head=int(model_cfg["n_head"]),
            n_kv_head=int(model_cfg.get("n_kv_head", model_cfg["n_head"])),
            ffn_dim=int(model_cfg.get("ffn_dim", int(8 / 3 * int(model_cfg["n_embd"])))),
            dropout=float(model_cfg["dropout"]),
            eps=float(model_cfg.get("layer_norm_epsilon", 1e-5)),
            rope_base=float(model_cfg.get("rope_base", 10000.0)),
        )
    else:
        raise ValueError(f"Unknown architecture '{arch}' in checkpoint.")
