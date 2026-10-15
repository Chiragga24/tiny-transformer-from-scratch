from __future__ import annotations

import math

import torch
from torch import nn


class Linear(nn.Module):
    """A bias-free linear transformation implemented without ``nn.Linear``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        # nn.Module sets up the machinery that discovers parameters, moves them
        # between devices, and saves/loads them in a state_dict.
        super().__init__()

        # Store W in the conventional PyTorch layout (d_out, d_in). Wrapping the
        # tensor in nn.Parameter tells autograd and optimizers that it is trainable.
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        # The reference architecture specifies a Xavier-like normal initialization with
        # variance 2 / (d_in + d_out), truncated at three standard deviations.
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x ends in d_in, while weight is (d_out, d_in), so transpose weight to
        # make the contracting dimensions align. Leading batch dimensions remain.
        return x @ self.weight.T


class Embedding(nn.Module):
    """Map integer token IDs to learned vectors without using ``nn.Embedding``."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # Each vocabulary item owns one learnable vector. The batch size and
        # sequence length are not stored here; they come from the input IDs.
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        # torch.empty only allocates memory. Fill it with the initialization
        # used by the reference architecture: N(0, 1), truncated to [-3, 3].
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Advanced indexing selects one row per token ID. If token_ids has shape
        # (batch, sequence), the result has shape (batch, sequence, embedding_dim).
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Normalize vectors by their root-mean-square magnitude."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps

        # RMSNorm has a learned multiplicative scale but no additive bias.
        # Ones make the scale neutral at initialization.
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype

        # Calculate the normalization in float32 to avoid overflow or precision
        # loss when model activations use float16 or bfloat16.
        x_float = x.to(torch.float32)
        mean_square = x_float.square().mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(mean_square + self.eps)

        # The d_model-sized weight broadcasts across batch and sequence axes.
        return (normalized * self.weight).to(input_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    """Apply SiLU elementwise without using ``torch.nn.functional.silu``."""
    # SiLU keeps large positive values, suppresses large negative values, and
    # remains smooth everywhere: SiLU(x) = x * sigmoid(x).
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """The gated feed-forward network used inside each Transformer block."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # w1 and w3 form two parallel branches with identical output shapes.
        # This allows their results to be multiplied element by element.
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

        # After gating in the wider d_ff space, project back to d_model so the
        # result has the same final dimension as the Transformer input.
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = W2(SiLU(W1(x)) * W3(x)). The multiplication is elementwise.
        activated_values = silu(self.w1(x))
        gate_values = self.w3(x)
        return self.w2(activated_values * gate_values)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute a numerically stable softmax along one dimension."""
    # Subtracting the maximum does not change softmax probabilities, but it
    # prevents exp(x) from overflowing when logits contain large values.
    shifted = x - x.max(dim=dim, keepdim=True).values
    exponentials = torch.exp(shifted)
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return attention-weighted values for queries, keys, and values."""
    d_k = Q.shape[-1]

    # Transpose only K's final two dimensions. This computes every query-key
    # dot product while preserving optional batch and attention-head dimensions.
    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        # A False entry represents a forbidden query-key connection. Negative
        # infinity makes its probability exactly zero after softmax.
        scores = scores.masked_fill(~mask, float("-inf"))

    # Normalize across keys: each query gets a probability distribution over
    # the values it is allowed to read.
    attention_weights = softmax(scores, dim=-1)

    # Combine value vectors according to those probabilities. The result keeps
    # the query count but ends in the value dimension d_v.
    return attention_weights @ V


class RotaryPositionalEmbedding(nn.Module):
    """Apply position-dependent 2D rotations to adjacent feature pairs."""

    def __init__(
        self,
        d_k: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires an even feature dimension")

        # Pair i rotates at frequency theta^(-2i/d_k). Combining every token
        # position with every frequency gives one angle per position and pair.
        pair_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inverse_frequencies = theta ** (-pair_indices / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * inverse_frequencies[None, :]

        # These tables are derived constants, not learned parameters. Registering
        # buffers makes them follow the module between devices without optimizing them.
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        positions = token_positions.to(device=self.cos_cache.device, dtype=torch.long)
        cos = self.cos_cache[positions].to(dtype=x.dtype)
        sin = self.sin_cache[positions].to(dtype=x.dtype)

        # Attention tensors include a head dimension that token_positions does
        # not. Add singleton dimensions before sequence so broadcasting covers it.
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        # Treat adjacent coordinates as (x, y) points in independent 2D planes.
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        # Interleave the rotated even and odd coordinates back into the original
        # final dimension: (0, 1), (2, 3), and so on.
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class MultiHeadSelfAttention(nn.Module):
    """Causal self-attention computed in parallel across multiple heads."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Each projection computes all heads at once. We split the resulting
        # d_model dimension into (num_heads, head_dim) inside forward.
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if (theta is None) != (max_seq_len is None):
            raise ValueError("theta and max_seq_len must either both be provided or both be omitted")
        self.rope = (
            RotaryPositionalEmbedding(self.head_dim, theta, max_seq_len, device=device)
            if theta is not None and max_seq_len is not None
            else None
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[-2]
        leading_dimensions = x.shape[:-2]

        # (..., sequence, d_model) -> (..., sequence, heads, head_dim)
        x = x.reshape(*leading_dimensions, sequence_length, self.num_heads, self.head_dim)

        # Put heads before sequence so attention treats each head as an
        # independent leading/batch dimension.
        return x.transpose(-3, -2)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sequence_length = x.shape[-2]
        leading_dimensions = x.shape[:-2]

        queries = self._split_heads(self.q_proj(x))
        keys = self._split_heads(self.k_proj(x))
        values = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            # Position changes query-key similarity, while values remain the
            # information retrieved after attention decides where to look.
            queries = self.rope(queries, token_positions)
            keys = self.rope(keys, token_positions)

        # Row i represents query position i. It may read keys 0 through i, but
        # positions above the diagonal are future tokens and must stay hidden.
        causal_mask = torch.tril(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device)
        )
        attended = scaled_dot_product_attention(queries, keys, values, causal_mask)

        # (..., heads, sequence, head_dim) -> (..., sequence, heads, head_dim)
        # -> (..., sequence, d_model). reshape safely handles the transposed tensor.
        joined = attended.transpose(-3, -2).reshape(
            *leading_dimensions, sequence_length, self.d_model
        )
        return self.output_proj(joined)


class TransformerBlock(nn.Module):
    """A LLaMA-style pre-norm decoder block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # Each sublayer receives its own normalization and learned scale.
        self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm attention: normalize first, run causal attention, then add the
        # original input through the residual path.
        z = x + self.attn(self.ln1(x), token_positions)

        # The second residual starts from z (not the original x). SwiGLU is the
        # complete feed-forward sublayer, so no separate FFN wrapper is needed.
        return z + self.ffn(self.ln2(z))


class TransformerLM(nn.Module):
    """A decoder-only Transformer that predicts logits for the next token."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.context_length = context_length

        # Integer token IDs become d_model-dimensional residual-stream vectors.
        self.token_embeddings = Embedding(
            vocab_size,
            d_model,
            device=device,
            dtype=dtype,
        )

        # ModuleList registers every block and gives state-dict names such as
        # layers.0.attn.q_proj.weight, layers.1.ffn.w1.weight, and so on.
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    eps=eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # Normalize the final residual stream, then project each token vector to
        # one unnormalized score (logit) for every vocabulary item.
        self.ln_final = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length {self.context_length}"
            )

        hidden_states = self.token_embeddings(token_ids)
        token_positions = torch.arange(sequence_length, device=token_ids.device)

        # Every block updates the same residual stream. Passing positions makes
        # shorter-than-context inputs use precisely the positions they contain.
        for layer in self.layers:
            hidden_states = layer(hidden_states, token_positions)

        hidden_states = self.ln_final(hidden_states)
        return self.lm_head(hidden_states)
