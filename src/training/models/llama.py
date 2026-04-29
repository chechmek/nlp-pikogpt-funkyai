"""LLaMA-style decoder-only transformer (RoPE, RMSNorm, SwiGLU, GQA)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (as in LLaMA)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._cache_len = 0
        self.register_buffer("_cos_cache", torch.empty(0), persistent=False)
        self.register_buffer("_sin_cache", torch.empty(0), persistent=False)

    def _update_cache(self, seq_len: int, device: torch.device) -> None:
        if seq_len <= self._cache_len:
            return
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        self._cos_cache = emb.cos()
        self._sin_cache = emb.sin()
        self._cache_len = seq_len

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_cache(seq_len, device)
        return self._cos_cache[:seq_len], self._sin_cache[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # x: (B, n_heads, T, head_dim), cos/sin: (T, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class GQAAttention(nn.Module):
    """Grouped-Query Attention with RoPE (LLaMA-style)."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.n_rep = n_head // n_kv_head

        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * self.head_dim, n_embd, bias=False)
        self.attn_dropout = dropout

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        # Repeat KV heads for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention (uses Flash Attention when available)
        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (LLaMA-style)."""

    def __init__(self, n_embd: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, ffn_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    """Single LLaMA transformer block (pre-norm with RMSNorm)."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int,
        ffn_dim: int,
        dropout: float,
        eps: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(n_embd, eps)
        self.attn = GQAAttention(n_embd, n_head, n_kv_head, dropout)
        self.ffn_norm = RMSNorm(n_embd, eps)
        self.ffn = SwiGLUFFN(n_embd, ffn_dim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.resid_dropout(self.attn(self.attn_norm(x), cos, sin))
        x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))
        return x


class LlamaModel(nn.Module):
    """
    LLaMA-style decoder-only language model.

    Features: RMSNorm, RoPE, SwiGLU FFN, Grouped-Query Attention, no bias.
    Compatible interface with CausalTransformerLM (same forward signature).
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        n_kv_head: int | None = None,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        eps: float = 1e-5,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        if n_kv_head is None:
            n_kv_head = n_head
        if ffn_dim is None:
            # LLaMA default: 8/3 * n_embd, rounded to nearest multiple of 64
            ffn_dim = int(2 * (4 * n_embd) / 3)
            ffn_dim = ((ffn_dim + 63) // 64) * 64

        head_dim = n_embd // n_head

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.rotary = RotaryEmbedding(head_dim, max_seq_len, base=rope_base)

        self.layers = nn.ModuleList([
            LlamaBlock(n_embd, n_head, n_kv_head, ffn_dim, dropout, eps)
            for _ in range(n_layer)
        ])
        self.norm = RMSNorm(n_embd, eps)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        B, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds model max_seq_len {self.max_seq_len}"
            )

        x = self.token_embedding(input_ids)
        cos, sin = self.rotary(T, input_ids.device)

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss: torch.Tensor | None = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {"logits": logits, "loss": loss}
