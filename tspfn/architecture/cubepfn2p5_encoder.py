from __future__ import annotations
import os
from typing import Tuple, Union, Literal
from shutil import copy2
from pathlib import Path

from abc import ABC
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import types
from tabpfn.checkpoint import Checkpoint
from tabpfn.architectures import ARCHITECTURES
from tabpfn.model_loading import load_model_criterion_config, load_model, _load_checkpoint_cached
from tspfn.architecture.pe_utils import rope_compute_heads_wrapper, interpolate_pos_encoding
from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast
from typing_extensions import override

import pydantic
import torch.utils.checkpoint
from torch.nn.attention import SDPBackend, sdpa_kernel

from tabpfn.architectures.encoders.steps._ops import (
    select_features,
    torch_nanmean,
)
from tabpfn.architectures.interface import (
    Architecture,
    ArchitectureConfig,
    PerformanceOptions,
)
from tabpfn.architectures.shared.attention_gqa_check import gqa_is_supported
from tabpfn.architectures.shared.chunked_evaluate import chunked_evaluate_maybe_inplace
from tabpfn.architectures.shared.column_embeddings import load_column_embeddings
from tabpfn.preprocessing.torch.torch_standard_scaler import TorchStandardScaler

NAN_INDICATOR = -2.0
INFINITY_INDICATOR = 2.0
NEG_INFINITY_INDICATOR = 4.0

# Feature group size multiplier to compute the size of the encodings projected
# to the embedding dimension. Since we add nan indicators we multiply the
# number of features groups by 2.
ENCODING_SIZE_MULTIPLIER = 2

if TYPE_CHECKING:
    from tabpfn.constants import TaskType

import logging as _logging
_logger = _logging.getLogger(__name__)


"""The TabPFN v2.5 architecture.

Compared to v2, this adds thinking rows, supports different input encoders, and reduces
the MLP hidden layer size.

Note that this version does not support a KV cache.

Copyright (c) Prior Labs GmbH 2026.
"""


@pydantic.dataclasses.dataclass
class TabPFNV2p5Config(ArchitectureConfig):
    """Configuration for the single-file TabPFN v2.5 architecture."""

    name: str = "TabPFN-v2.5"
    emsize: int = 192
    nlayers: int = 24
    nhead: int = 3
    """Number of key/value heads to use for per-column-inter-row attention."""

    features_per_group: int = 3
    """If > 1, the features will be grouped into groups of this size and the attention
    is across groups."""

    num_thinking_rows: int = 64
    """Number of "thinking rows" to prepend to each dataset, see AddThinkingRows."""

    encoder_type: Literal["linear", "mlp"] = "linear"
    """Whether to use a linear or MLP encoder in the input encoder."""

    encoder_mlp_hidden_dim: int = 1024
    """Hidden dimension for the MLP embedder."""


def apply_rope(
    t: Tensor,
    inv_freq: Tensor,
    *,
    interleaved: bool = False,
) -> Tensor:
    """Apply rotary positional embeddings to ``t`` along seq_dim=-2.

    All intermediate math is done in ``inv_freq.dtype`` (fp32 by
    construction) and the result is cast back to ``t.dtype``.

    Args:
        t: Tensor of shape ``(..., S, D)`` where the head dim ``D`` is
            even. The sequence dim is the second-to-last axis.
        inv_freq: ``(D // 2,)`` inverse frequencies (typically
            ``1 / theta ** (2i / D)``).
        interleaved: When ``True``, rotates dimension pairs
            ``(0, 1), (2, 3), …`` (LLaMA/HF interleaved layout). When
            ``False`` (default), splits the last dim into two contiguous
            halves and rotates them against each other.
    """
    dtype = t.dtype
    seq_len = t.shape[-2]
    positions = torch.arange(seq_len, device=t.device, dtype=inv_freq.dtype)
    freqs = positions[:, None] * inv_freq[None, :]  # (S, D/2)
    cos = freqs.cos()
    sin = freqs.sin()
    if interleaved:
        cos = cos.repeat_interleave(2, dim=-1)  # (S, D)
        sin = sin.repeat_interleave(2, dim=-1)
        t_even = t[..., 0::2]
        t_odd = t[..., 1::2]
        # stack → (..., D/2, 2) rows (-t_odd, t_even); flatten → (-t1, t0, -t3, t2, …)
        t_rotated = torch.stack((-t_odd, t_even), dim=-1).flatten(-2)
    else:
        cos = torch.cat((cos, cos), dim=-1)  # (S, D)
        sin = torch.cat((sin, sin), dim=-1)
        half = t.shape[-1] // 2
        t_rotated = torch.cat((-t[..., half:], t[..., :half]), dim=-1)
    return (t * cos + t_rotated * sin).to(dtype)


class RotaryEmbedding(nn.Module):
    """Compile-friendly rotary positional embedding.

    Args:
        dim: Per-head rotation dimension. Must be even.
        theta: Base for the rotary frequencies (10_000 in the original
            paper, 100_000 in our configs).
        interleaved: See :func:`apply_rope`.
    """

    def __init__(
        self,
        dim: int,
        *,
        theta: float = 10_000.0,
        interleaved: bool = False,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, f"RoPE head dim must be even, got {dim}"
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        # Store as a non-learnable nn.Parameter (not a buffer) to match the
        # upstream RotaryEmbedding which has ``self.freqs = nn.Parameter(...,
        # requires_grad=False)``. This preserves the parameter count seen by
        # the optimizer, avoiding subtle numerical drift in training due to
        # Adam state ordering changes.
        self.freqs = nn.Parameter(inv_freq, requires_grad=False)
        self.interleaved = interleaved

    def rotate_queries_or_keys(self, t_BHSD: Tensor) -> Tensor:
        """Apply RoPE to t_BSHD."""
        return apply_rope(t_BHSD, self.freqs, interleaved=self.interleaved)


class _DtypeMatchingRMSNorm(nn.RMSNorm):
    """RMSNorm that casts weight to match the input dtype.

    Fused CUDA kernels require matching dtypes; casting the tiny weight/bias per-call
    avoids unfused fallbacks under autocast.
    """

    @override
    def forward(self, input: Tensor) -> Tensor:
        if self.weight is not None and self.weight.dtype != input.dtype:
            return F.rms_norm(
                input,
                self.normalized_shape,
                self.weight.to(input.dtype),
                self.eps,
            )
        return super().forward(input)


class SoftmaxScalingMLP(nn.Module):
    """Query-aware attention scaling using MLPs to compute scaling factors.

    Applies scaling to queries:

    q_scaled = q * base_mlp(logn) * (1 + tanh(query_mlp(q))),

    where the base MLP learns length-dependent scaling and the query MLP
    learns query-dependent modulation.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        n_hidden: int = 64,
    ):
        """Initializes the SoftmaxScalingMLP module.

        Args:
            num_heads: Number of attention heads.
            head_dim: Dimension of each attention head.
            n_hidden: Number of hidden units in the MLPs.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        base_out_dim = num_heads * head_dim
        query_out_dim = head_dim

        self.base_mlp = nn.Sequential(nn.Linear(1, n_hidden), nn.GELU(), nn.Linear(n_hidden, base_out_dim))
        self.query_mlp = nn.Sequential(nn.Linear(head_dim, n_hidden), nn.GELU(), nn.Linear(n_hidden, query_out_dim))
        # ensures initial modulation is zero
        nn.init.zeros_(self.query_mlp[-1].weight)  # type: ignore
        nn.init.zeros_(self.query_mlp[-1].bias)  # type: ignore

    @override
    def forward(self, q_BSHD: Tensor, n: int) -> Tensor:
        """Applies scalable attention scaling to queries.

        Args:
            q_BSHD: Query tensor after projection, shape `[B, S, H, D]`.
                B: Batch size.
                S: Sequence length.
                H: Number of heads.
                D: Head dimension.
            n: Number of elements for log-n scaling.

        Returns:
            Scaled query tensor, same shape as `q_BSHD`.
        """
        logn_11 = _safe_log_seqlen(n, q_BSHD.device, q_BSHD.dtype).reshape(1, 1)
        base_scales = self.base_mlp(logn_11).view(1, 1, self.num_heads, self.head_dim)
        modulation = 1 + torch.tanh(self.query_mlp(q_BSHD))
        scales = base_scales * modulation
        return q_BSHD * scales


class ManyClassDecoder(nn.Module):
    """Attention-based retrieval decoder for many-class classification.

    Computes weighted (by attention score) average over one-hot encoded
    train targets, then takes the log to obtain logits.  Supports arbitrary
    class counts by chunking the value (one-hot) dimension into head_dim-sized
    pieces and folding them into the batch dimension for a single flash-attention
    call.
    """

    def __init__(
        self,
        max_num_classes: int,
        input_size: int,
        head_dim: int = 64,
        num_heads: int = 6,
        softmax_scaling_layer: nn.Module | None = None,
    ):
        """Init."""
        super().__init__()
        self.max_num_classes = max_num_classes
        self.input_size = input_size
        self.attention_size = head_dim * num_heads
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.q_projection = nn.Linear(self.input_size, self.attention_size)
        self.k_projection = nn.Linear(self.input_size, self.attention_size)
        self.softmax_scaling_layer = softmax_scaling_layer

    @override
    def forward(
        self,
        train_embeddings: Tensor,  # (B, N, E)
        test_embeddings: Tensor,  # (B, M, E)
        targets: Tensor,  # (B, N) - class indices
    ) -> Tensor:
        """Perform a forward pass."""
        B, M, _ = test_embeddings.shape
        q_BME = self.q_projection(test_embeddings)
        # Mirrors the dtype guard in ICLAttention's cached path.
        if train_embeddings.dtype != q_BME.dtype:
            train_embeddings = train_embeddings.to(q_BME.dtype)
        k_BNE = self.k_projection(train_embeddings)

        if M == 0:
            # OOM checks at training start run with no test rows. Flash attention
            # rejects a query sequence of length 0, so we return early.
            # Both dummy terms keep the output in the computation graph so that
            # gradients flow through both projections during memory estimation.
            empty = test_embeddings.new_empty((0, B, self.max_num_classes))
            return empty + (q_BME.sum() + k_BNE.sum()) * 0.0

        # WARNING Change the original code to compute num_classes_in_batch from targets
        # to adapt to the new setting where we may have fewer classes in a batch than max_num_classes
        num_classes_in_batch = targets.max().item() + 1
        one_hot_targets_BNT = (
            # F.one_hot(targets.long(), num_classes=self.max_num_classes)
            F.one_hot(targets.long(), num_classes=num_classes_in_batch)
            .to(dtype=q_BME.dtype)
            .contiguous()
        )

        q_BMHD = q_BME.view(B, M, self.num_heads, self.head_dim).contiguous()
        k_BNHD = k_BNE.view(B, -1, self.num_heads, self.head_dim).contiguous()
        one_hot_targets_BNHT = one_hot_targets_BNT.unsqueeze(2).expand(-1, -1, self.num_heads, -1).contiguous()
        test_output_BMHT = _chunked_class_attention(
            q_BMHD,
            k_BNHD,
            one_hot_targets_BNHT,
            softmax_scaling_layer=self.softmax_scaling_layer,
        )
        test_output_BMT = test_output_BMHT.mean(2)  # average over heads

        test_output_MBT = test_output_BMT.transpose(0, 1)
        # convert to logits:
        return torch.log(torch.clamp(test_output_MBT, min=1e-5) + 3e-5)


def _chunked_class_attention(
    q_BSHD: Tensor,
    k_BJHD: Tensor,
    v_BJHT: Tensor,
    softmax_scaling_layer: nn.Module | None = None,
) -> Tensor:
    """Run retrieval attention where the value dimension C may exceed head_dim D.

    Splits V into head_dim-sized chunks along the class axis, folds the chunk
    index into the batch dimension, and dispatches a single flash-attention call.
    This avoids the O(N*M) memory cost of the math backend for any class count.

    Args:
        q_BSHD: Query tensor of shape (B, S, H, D) for test points.
        k_BJHD: Key tensor of shape (B, J, H, D) for train points.
        v_BJHT: Value tensor of shape (B, J, H, T) holding one-hot class
            encodings; T may be larger than D.
        softmax_scaling_layer: Optional scaling module to scale queries before SDPA.

    Returns:
        Output tensor of shape (B, S, H, T).
    """
    B, S, H, D = q_BSHD.shape
    T = v_BJHT.shape[-1]
    num_chunks = math.ceil(T / D)

    # Pad V to a multiple of D along the class axis
    pad = num_chunks * D - T
    if pad > 0:
        v_BJHT = F.pad(v_BJHT, (0, pad))

    # Fold chunk index into batch dimension
    J = v_BJHT.shape[1]
    v_folded = (
        v_BJHT.reshape(B, J, H, num_chunks, D).permute(0, 3, 1, 2, 4).reshape(B * num_chunks, J, H, D).contiguous()
    )
    q_folded = q_BSHD.unsqueeze(1).expand(-1, num_chunks, -1, -1, -1).reshape(B * num_chunks, S, H, D).contiguous()
    k_folded = k_BJHD.unsqueeze(1).expand(-1, num_chunks, -1, -1, -1).reshape(B * num_chunks, J, H, D).contiguous()

    # Single flash-attention call across all chunks
    out_folded = _batched_scaled_dot_product_attention(
        q_folded, k_folded, v_folded, softmax_scaling_layer=softmax_scaling_layer
    )

    # Unfold and trim padding: (B*K, S, H, D) -> (B, S, H, T)
    return out_folded.reshape(B, num_chunks, S, H, D).permute(0, 2, 3, 1, 4).reshape(B, S, H, num_chunks * D)[..., :T]


class AddThinkingRows(nn.Module):
    """Takes the embedded input and prepends "thinking rows" to it.

    Adjusts the single_eval_pos appropriately to account for the new, longer input.

    The thinking tokens give the model more computational capacity to perform in-context
    learning.
    """

    def __init__(self, num_thinking_rows: int, embedding_size: int) -> None:
        super().__init__()
        self.num_thinking_rows = num_thinking_rows
        # We have to work with variable numbers of features, so we use the same token
        # for each feature.
        self.row_token_values_TE = nn.Parameter(
            torch.empty(num_thinking_rows, embedding_size)
        )
        self.reset_parameters()

    @override
    def forward(
        self,
        x_BRiCLD: torch.Tensor,
        single_eval_pos: int,
    ) -> tuple[torch.Tensor, int]:
        """Prepends the thinking tokens to the embedded input.

        Args:
            x_BRiCLD: Input tensor of shape (B, Ri, C, E), where:
                - B: batch size
                - Ri: number of input rows (train and test)
                - C: number of channels
                - L: number of features per channel (T + 1 for the target)
                - D: Model embedding size.
            single_eval_pos: Rows after this index are treated as evaluation rows.

        Returns:
            A tuple, (x_BRCLD, updated single_eval_pos), where where x_BRCLD has added
            rows, now having shape (B, R = Ri + num thinking rows, C, D).
        """
        # T: num thinking rows
        # R: Ri + T, total rows including thinking rows
        batch_size, _, num_channels, num_features, _ = x_BRiCLD.shape
        thinking_rows_BTCLE = (
            self.row_token_values_TE.unsqueeze(0)
            .unsqueeze(2)
            .expand(batch_size, -1, num_channels, num_features, -1)
        )
        x_BRCLD = torch.cat([thinking_rows_BTCLE, x_BRiCLD], dim=1)
        single_eval_pos += self.num_thinking_rows
        return x_BRCLD, single_eval_pos

    def reset_parameters(self) -> None:
        """Set the tokens to randomly-sampled values."""
        # This is the initialisation used in torch.nn.Embedding, so hopefully a
        # reasonable choice for our application.
        torch.nn.init.normal_(self.row_token_values_TE)


class Attention(nn.Module):
    """Base class for the between-features and between-rows attention layers."""

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
        use_rope: bool = True,
        rope_base: float = 100_000,
    ):
        """Construct a new instance.

        Args:
            embedding_size: The size of the input embedding.
            num_heads: The number of heads to use.
            head_dim: The dimensionality of the query, key and value vectors.
            device: The device to use for the layer parameters.
            dtype: The data type to use for the layer parameters.
            use_rope: Whether to use rotary positional embeddings.
            rope_base: The base for the rotary positional embeddings.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        device_and_dtype_no_bias = {"device": device, "dtype": dtype, "bias": False}

        self.rope = (
            RotaryEmbedding(
                dim=embedding_size // num_heads,
                theta=int(rope_base),
                interleaved=False,
            )
            if use_rope
            else None
        )

        self.q_projection = nn.Linear(
            embedding_size, head_dim * num_heads, **device_and_dtype_no_bias
        )
        self.k_projection = nn.Linear(
            embedding_size, head_dim * num_heads, **device_and_dtype_no_bias
        )
        self.v_projection = nn.Linear(
            embedding_size, head_dim * num_heads, **device_and_dtype_no_bias
        )

        self.out_projection = nn.Linear(
            head_dim * num_heads, embedding_size, **device_and_dtype_no_bias
        )

        torch.nn.init.xavier_uniform_(self.q_projection.weight)
        torch.nn.init.xavier_uniform_(self.k_projection.weight)
        torch.nn.init.xavier_uniform_(self.v_projection.weight)
        torch.nn.init.zeros_(self.out_projection.weight)


class AlongRowAttention(Attention):
    """Computes the attention between features of a single row.

    This is standard multi-head self-attention, where all features attend to each other.
    """

    @override
    def forward(self, x_BrSE: torch.Tensor) -> torch.Tensor:
        """Forward pass for along-row attention between features.

        Args:
            x_BrSE: The input tensor of shape (Br, C, E), where:
                - Br: Batch size * num rows.
                - C: Number of feature groups.
                - E: Embedding size.
        """
        # H: number of heads.
        # D: head dimension.
        # F: head_dimension * number of heads.
        Br, C, _ = x_BrSE.shape
        q_flat_BrCHF = self.q_projection(x_BrSE)
        k_flat_BrCHF = self.k_projection(x_BrSE)
        v_flat_BrCHF = self.v_projection(x_BrSE)
        q_BrCHD = q_flat_BrCHF.view(Br, C, -1, self.head_dim)
        k_BrCHD = k_flat_BrCHF.view(Br, C, -1, self.head_dim)
        v_BrCHD = v_flat_BrCHF.view(Br, C, -1, self.head_dim)

        if self.rope is not None:
            q_BrCHD = self.rope.rotate_queries_or_keys(q_BrCHD.transpose(1, 2)).transpose(1, 2)
            k_BrCHD = self.rope.rotate_queries_or_keys(k_BrCHD.transpose(1, 2)).transpose(1, 2)


        output_BrHCD = _batched_scaled_dot_product_attention(q_BrCHD, k_BrCHD, v_BrCHD)
        output_BrCF = output_BrHCD.reshape(Br, C, self.head_dim * self.num_heads)
        return self.out_projection(output_BrCF)


class AlongColumnAttention(Attention):
    """Computes the attention between cells of a single column.

    This is multi-head attention featuring:
    - An implicit mask: The training rows attend to each other and themselves, but not
        the test rows. The test rows only attend to the training rows, and not
        themselves. By not attending to themselves, this avoids the requirement for an
        explicit mask.
    - Multi-query attention for the test rows: All the query heads for the test rows
        attend to the first key-value head. This is a further optimisation that only
        requires including one head in the key-value cache.
    """

    @override
    def forward(
        self,
        x_BcRE: torch.Tensor,
        single_eval_pos: int | None = None,
    ) -> torch.Tensor:
        """Forward pass for attention between cells of a single column.

        Args:
            x_BcRE: The input tensor of shape (Bc, R, E), where:
                - Bc: Batch size * number of columns
                - R: Total rows (thinking + train + test).
                - E: Embedding size.
            single_eval_pos: The position from which on everything is treated as test
                set. If None, no mask is applied and all positions are attended to. If
                given, each query after single_eval_pos will only attend to positions
                before single_eval_pos.
        """
        # H: number of heads.
        # D: head dimension.
        # F: head_dimension * number of heads.
        # N: number of thinking and train rows = single_eval_pos
        # M: number of test rows
        Bc, R, _ = x_BcRE.shape
        # If no single_eval_pos was specified, then the whole input is training.
        N = R if single_eval_pos is None else single_eval_pos

        q_flat_BcSHF = self.q_projection(x_BcRE)
        k_flat_BcNHF = self.k_projection(x_BcRE[:, :N])
        v_flat_BcNHF = self.v_projection(x_BcRE[:, :N])
        q_BcRHD = q_flat_BcSHF.view(Bc, R, -1, self.head_dim)
        k_BcNHD = k_flat_BcNHF.view(Bc, N, -1, self.head_dim)
        v_BcNHD = v_flat_BcNHF.view(Bc, N, -1, self.head_dim)

        if single_eval_pos == R:
            output_BcSHD = _batched_scaled_dot_product_attention(
                q_BcRHD, k_BcNHD, v_BcNHD
            )
        else:
            out_train_BcNHD = _batched_scaled_dot_product_attention(
                q_BcRHD[:, :N], k_BcNHD, v_BcNHD
            )
            out_test_BcMHD = _batched_scaled_dot_product_attention(
                q_BcRHD[:, N:], k_BcNHD[:, :, :1], v_BcNHD[:, :, :1]
            )
            output_BcSHD = torch.cat([out_train_BcNHD, out_test_BcMHD], dim=1)

        output_BcSF = output_BcSHD.reshape(Bc, R, self.head_dim * self.num_heads)
        return self.out_projection(output_BcSF)


def _safe_log_seqlen(n: int | Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Compute log(n) safely, avoiding fp16 overflow for large `n`."""
    if isinstance(n, Tensor):
        return n.to(torch.float32).clamp(min=1).log().to(dtype)
    # Materialise `n` via arithmetic on a 0-d tensor rather than
    # `torch.as_tensor(n, ...)`. The latter bakes `n` into the graph as a constant and
    # emits a `n == <value>` guard, triggering a recompile on every new value
    # `one * n` keeps the value symbolic when `n` is a SymInt.
    one = torch.ones((), dtype=torch.float32, device=device)
    return (one * n).clamp(min=1).log().to(dtype)

def _batched_scaled_dot_product_attention(
    q_BSHD: torch.Tensor, k_BSJD: torch.Tensor, v_BSJD: torch.Tensor
) -> torch.Tensor:
    """Execute scaled dot product attention, chunked over the batch dimension.

    Our between-feature attention can have a large batch size.
    E.g., for 2048 datapoints, a batch size of 32, and 6 heads,
    we compute 2048 * 32 * 6 = 393216 attentions.
    This is larger than the maximum launch grid size of cuda and will raise an error.
    Thus, we split the inputs into chunks of the maximum batch size, and execute these
    sequentially.
    """
    q_BHSD = q_BSHD.permute(0, 2, 1, 3)
    k_BJSD = k_BSJD.permute(0, 2, 1, 3)
    v_BJSD = v_BSJD.permute(0, 2, 1, 3)

    # In the case of multi-query attention, the keys and values will have only one head.
    # GQA is only supported with fp16/bf16 dtypes - the fused attention kernels
    # don't support GQA with float32.
    dtype_supports_gqa = q_BHSD.dtype in {torch.float16, torch.bfloat16}
    if gqa_is_supported() and dtype_supports_gqa:
        keys = k_BJSD
        values = v_BJSD
        enable_gqa = {"enable_gqa": True}
    else:
        # On older GPUs or with float32 dtype, the fused attention kernels don't
        # support broadcasting, so we manually expand the keys and values to the
        # same number of heads as the queries.
        keys = k_BJSD.expand(-1, q_BHSD.shape[-3], -1, -1)
        values = v_BJSD.expand(-1, q_BHSD.shape[-3], -1, -1)
        enable_gqa = {}

    # Enable backends explicitly. MATH is included as a last resort since it
    # stores attention scores explicitly and uses more memory, but we prefer
    # a slow fallback over a hard crash (e.g. on T4 GPUs on github runners where
    # flash/efficient attention kernels may not support all configurations).
    backends = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.MATH,
    ]
    num_parallel_calls = q_BHSD.shape[:2].numel()
    CUDA_MAX_GRID = 65536
    num_iterations = (num_parallel_calls + CUDA_MAX_GRID - 1) // CUDA_MAX_GRID
    sub_batch = (q_BHSD.shape[0] + num_iterations - 1) // num_iterations

    with sdpa_kernel(backends=backends):
        outputs = []
        for i in range(num_iterations):
            outputs.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q_BHSD[i * sub_batch : (i + 1) * sub_batch],
                    keys[i * sub_batch : (i + 1) * sub_batch],
                    values[i * sub_batch : (i + 1) * sub_batch],
                    attn_mask=None,
                    **enable_gqa,
                )
            )
    output_BHSD = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
    return output_BHSD.permute(0, 2, 1, 3)


class LowerPrecisionLayerNorm(torch.nn.LayerNorm):
    """LayerNorm that maintains FP16 precision in autocast mode.

    PyTorch autocast runs LayerNorm in FP32, which has bad effects on our performance
    (we observed 2x slower) and uses more memory. This layer instead disabled autocast
    for the layer norm, so FP16 is maintained if this is the input format.

    WARNING: this could lead to instabilities for larger hidden sizes, so we only enable
    it for hidden sizes of <512.
    """

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if (
            input.dtype in (torch.float16, torch.bfloat16)
            and sum(self.normalized_shape) < 512
        ):
            with torch.amp.autocast("cuda" if input.is_cuda else "cpu", enabled=False):
                return super().forward(input)

        return super().forward(input)


class TabPFNBlock(nn.Module):
    """A block of one column-wise, one row-wise attention layer and an MLP layer."""

    def __init__(
        self,
        *,
        emsize: int,
        nhead: int,
        dim_feedforward: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
        use_rope: bool = True,
    ) -> None:
        """TabPFNBlock constructor.

        Args:
            emsize: The input embedding size.
            nhead: The number of query attention heads to use.
            dim_feedforward: The dimensionality of the feedforward network.
            device: The device to use for the layer parameters.
            dtype: The data type to use for the layer parameters.
        """
        super().__init__()
        device_and_dtype = {"device": device, "dtype": dtype}
        assert emsize % nhead == 0
        # The features of a single sample attend to each other.
        self.per_sample_attention_between_features = AlongRowAttention(
            embedding_size=emsize,
            num_heads=nhead,
            head_dim=emsize // nhead,
            use_rope=use_rope,
            **device_and_dtype,
        )

        # The cells of a single column attend to each other.
        self.per_column_attention_between_cells = AlongColumnAttention(
            embedding_size=emsize,
            num_heads=nhead,
            head_dim=emsize // nhead,
            **device_and_dtype,
        )

        layer_norm_args = {**device_and_dtype, "elementwise_affine": False}
        self.layernorm_mha1 = LowerPrecisionLayerNorm(emsize, **layer_norm_args)
        self.layernorm_mha2 = LowerPrecisionLayerNorm(emsize, **layer_norm_args)
        self.layernorm_mlp = LowerPrecisionLayerNorm(emsize, **layer_norm_args)

        self.mlp = nn.Sequential(
            torch.nn.Linear(emsize, dim_feedforward, bias=False, **device_and_dtype),
            torch.nn.GELU(),
            torch.nn.Linear(dim_feedforward, emsize, bias=False, **device_and_dtype),
        )
        torch.nn.init.zeros_(cast("torch.nn.Linear", self.mlp[2]).weight)

    @override
    def forward(
        self,
        x_BRCE: torch.Tensor,
        single_eval_pos: int,
        save_peak_memory_factor: int | None,
    ) -> torch.Tensor:
        """Compute one column-wise, one row-wise attention, and an MLP layer.

        Uses post-norm.

        B: Batch size
        R: Number of rows / items
        C: Number of columns / features
        E: The embedding size of each cell.

        Args:
            x_BRCE:
                The transformer state passed as input to the layer of shape
                (batch_size, num_items, num_feature_blocks, d_model).
            single_eval_pos:
                The position from which on everything is treated as test set.
            save_peak_memory_factor:
                If not None, switch to the inference-optimised forward pass which
                reduces memory by chunking the evaluation of each layer over the batch
                dimension.
                If None, use the standard forward pass compatible with gradient
                computation.

        Returns:
            The transformed state
        """
        # -- First Block: Attention between features.
        x_BRCE = chunked_evaluate_maybe_inplace(
            self.per_sample_attention_between_features,
            x_BRCE,
            save_peak_memory_factor,
            residual=True,
            # The rows are folded into the batch, so computing attention over the column
            # here is per sample.
            batch_dims=2,
        )
        x_BRCE = chunked_evaluate_maybe_inplace(
            self.layernorm_mha1,
            x_BRCE,
            save_peak_memory_factor,
            residual=False,
            # The batch norm treats every token independently, so the batch includes
            # both the rows and the columns.
            batch_dims=3,
        )

        # -- Second Block: Attention between cells.
        # Call .contiguous() so that _chunk() can operate on x_BCRE in-place, when
        # memory saving is enabled.
        x_BCRE = x_BRCE.transpose(1, 2).contiguous()
        x_BCRE = chunked_evaluate_maybe_inplace(
            self.per_column_attention_between_cells,
            x_BCRE,
            save_peak_memory_factor,
            residual=True,
            # The columns are flattened into the batch, so we compute attention over the
            # cells of each column independently.
            batch_dims=2,
            single_eval_pos=single_eval_pos,
        )
        x_BCRE = chunked_evaluate_maybe_inplace(
            self.layernorm_mha2,
            x_BCRE,
            save_peak_memory_factor,
            residual=False,
            batch_dims=3,
        )
        # Again, call .contiguous() so that _chunk() can operate on x_BCRE in-place.
        x_BRCE = x_BCRE.transpose(1, 2).contiguous()

        # -- Third Block: MLP layer.
        x_BRCE = chunked_evaluate_maybe_inplace(
            self.mlp,
            x_BRCE,
            save_peak_memory_factor,
            residual=True,
            # The MLP also treats every token independently, so the batch includes both
            # the rows and the columns.
            batch_dims=3,
        )
        return chunked_evaluate_maybe_inplace(
            self.layernorm_mlp,
            x_BRCE,
            save_peak_memory_factor,
            residual=False,
            batch_dims=3,
        )


class TabPFNV2p5(Architecture):
    """TabPFN V2.5 with post-layernorm and self-attention on test-items."""

    def __init__(
        self,
        *,
        config: TabPFNV2p5Config,
        additional_config: dict[str, Any] | None = None,
        task_type: TaskType,
        max_n_out: int = 160,
        feature_positional_embedding: Literal["subspace"] | None = "subspace",
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        """Initializes the PerFeatureTransformer module.

        Args:
            config: The model hyperparameters.
            encoder: An InputEncoder, which takes a dictionary with tensors of shape
                [num_rows, batch_size, num_cols, features] and returns a single tensor
                of shape [num_rows, batch_size, input_size].
            task_type: The type of task the model should perform.
            max_n_out: The number of outputs the model should produce.
            feature_positional_embedding: The positional embedding type to use.
                The  positional embedding is added to the features to help the model
                distinguish them. Currently, only "subspace" is supported.
            device: The device to use for the layer parameters.
            dtype: The data type to use for the layer parameters.
        """
        super().__init__()
        if feature_positional_embedding != "subspace":
            raise ValueError("Currently only 'subspace' is supported.")
        self.input_size = config.emsize
        self.hidden_size = self.input_size * 2
        self.features_per_group = config.features_per_group
        self.max_n_out = max_n_out
        self.task_type: TaskType = task_type

        self.feature_group_embedder = self._get_feature_group_embedder(config)
        self.target_embedder = nn.Linear(ENCODING_SIZE_MULTIPLIER, config.emsize)
        self.add_thinking_rows = AddThinkingRows(
            num_thinking_rows=config.num_thinking_rows,
            embedding_size=config.emsize,
        )
        self.blocks = nn.ModuleList(
            TabPFNBlock(
                emsize=config.emsize,
                nhead=config.nhead,
                dim_feedforward=self.hidden_size,
                device=device,
                dtype=dtype,
                use_rope=additional_config.use_rope,
            )
            for _ in range(config.nlayers)
        )
        self.channel_blocks = nn.ModuleList(
            TabPFNBlock(
                emsize=config.emsize,
                nhead=config.nhead,
                dim_feedforward=self.hidden_size,
                device=device,
                dtype=dtype,
                use_rope=False,  
            )
            for _ in range(config.nlayers)
        )
        self.output_projection = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, max_n_out),
        )
        decoder_softmax_scaling = (
            SoftmaxScalingMLP(
                num_heads=additional_config.decoder_num_heads,
                head_dim=additional_config.decoder_head_dim,
                n_hidden=additional_config.softmax_scaling_mlp_hidden_dim,
            )
            if additional_config.decoder_use_softmax_scaling
            else None
        )
        self.many_class_decoder = ManyClassDecoder(
            max_num_classes=config.max_num_classes,
            input_size=config.emsize,
            head_dim=additional_config.decoder_head_dim,
            num_heads=additional_config.decoder_num_heads,
            softmax_scaling_layer=decoder_softmax_scaling,
        )
        self.standard_scaler = TorchStandardScaler()

        self.pre_generated_column_embeddings = load_column_embeddings()
        self.feature_positional_embedding_embeddings = nn.Linear(
            self.input_size // 4, self.input_size
        )
        self._do_encoder_nan_check = True
        self.emsize = config.emsize

    @property
    @override
    def embedding_dim(self) -> int:
        return self.emsize

    def _get_feature_group_embedder(self, config: TabPFNV2p5Config) -> nn.Module:
        """Get the feature group embedder."""
        encoding_size = config.features_per_group * ENCODING_SIZE_MULTIPLIER
        hidden_dim = config.encoder_mlp_hidden_dim
        emsize = config.emsize
        if config.encoder_type == "mlp":
            embedder = nn.Sequential(
                nn.Linear(encoding_size, hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim, emsize, bias=False),
            )
        else:
            embedder = nn.Linear(encoding_size, emsize, bias=False)
        return embedder

    def _add_column_embeddings(self, x_BRCGX: torch.Tensor) -> torch.Tensor:
        """Add a random embedding to each column to prevent feature collapse."""
        generator = torch.Generator(device=x_BRCGX.device).manual_seed(42)
        num_cols, encoding_size = x_BRCGX.shape[-2], x_BRCGX.shape[-1]
        embs = torch.randn(
            (num_cols, encoding_size // 4),
            device=x_BRCGX.device,
            dtype=x_BRCGX.dtype,
            generator=generator,
        )
        # Random numbers vary between different devices, even with a fixed seed. Thus we
        # use the pre-generated column embeddings where possible to ensure they are
        # consistent between pretraining and inference.
        # Some tests use a smaller embedding size, in which case we do not use the saved
        # embeddings.
        if embs.shape[1] == self.pre_generated_column_embeddings.shape[1]:
            embs[:2000] = self.pre_generated_column_embeddings[: embs.shape[0]].to(
                device=embs.device, dtype=embs.dtype
            )
        embs = self.feature_positional_embedding_embeddings(embs)
        x_BRCGX += embs[None, None, None]

        return x_BRCGX

    @override
    def forward(  # noqa: C901
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        y: torch.Tensor | dict[str, torch.Tensor] | None,
        ts_diff: torch.Tensor | None = None,
        *,
        only_return_standard_out: bool = True,
        categorical_inds: list[list[int]] | None = None,
        performance_options: PerformanceOptions | None = None,
        task_type: str | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Perform a forward pass.

        See ModelInterface.forward() for the full docstring.

        Args:
            x: Input features of shape (Ri, B, C, T) where:
                - Ri: number of input rows (train + test)
                - B: batch size
                - C: number of channels
                - T: time steps (for time series data)
            y: Input targets of shape (Ri, B) where:
                - Ri: number of input rows (train + test)
                - B: batch size

            only_return_standard_out: If True, only return the standard output of the model.
            categorical_inds: List of lists of indices for categorical features.
            performance_options: Options for controlling performance optimizations.
            task_type: The type of task to perform. If None, use the model's default
                task type.
        Returns:
            If only_return_standard_out is True, returns a tensor of shape (M, B, 1) where:
                - M: number of test samples
                - B: batch size
            If only_return_standard_out is False, returns a dictionary with the following keys:
                - "standard": Tensor of shape (M, B, 1) with the standard output of the model.
                - "train_embeddings": Tensor of shape (N, B, D) with the embeddings of the training samples, where:
                    - N: number of training samples
                    - B: batch size
                    - D: embedding size
                - "test_embeddings": Tensor of shape (M, B, D) with the embeddings of the test samples, where:
                    - M: number of test samples
                    - B: batch size
                    - D: embedding size
        """
        if performance_options is None:
            performance_options = self.get_default_performance_options()
        force_recompute_layer = performance_options.force_recompute_layer
        save_peak_memory_factor = performance_options.save_peak_memory_factor
        del categorical_inds
        if isinstance(x, dict):
            x = x["main"]
        if isinstance(y, dict):
            y = y["main"]
        if y is None:
            y = torch.zeros(0, device=x.device, dtype=x.dtype)

        if (
            not self.training
            and self.task_type == "multiclass"
            and (y > self.max_n_out - 1).any()
        ):
            raise ValueError(
                "Target is out of range. Make sure to use an ordinal encoded target. "
                f"Expected target values between 0 and {self.max_n_out - 1}, but got values"
                f" greater than {self.max_n_out - 1}."
            )

        # Ri = number of input rows (train + test, before adding thinking rows)
        # B = batch size
        # C = number of columns before grouping
        x_RiBCT = x
        num_train_rows, batch_size, num_channels, *_ = x_RiBCT.shape
        num_train_labels = y.shape[0]

        embedded_x_BRiCGX = self._preprocess_and_embed_features(
            x_RiBCT=x_RiBCT,
            num_train_labels=num_train_labels,
            batch_size=batch_size,
            num_channels=num_channels
        )

        embedded_y_BRiCX = self._preprocess_and_embed_targets(
            y=y,
            num_train_rows=num_train_rows,
            num_train_labels=num_train_labels,
            batch_size=batch_size,
            num_channels=num_channels
        )

        # Add the targets as an additional column.
        x_BRiCLD = torch.cat((embedded_x_BRiCGX, embedded_y_BRiCX[:, :, :, None]), dim=3)
        # This check results in a graph break with torch compile, so we only run it once
        # in the beginning and then disable it.
        if self._do_encoder_nan_check:
            if torch.isnan(x_BRiCLD).any():
                raise ValueError(
                    "Found NaNs in the encoded x and y. Make sure to use "
                    "a NaN-handling encoder."
                )
            self._do_encoder_nan_check = False

        # R = num rows + num thinking rows
        x_BRCLD, num_train_and_thinking_rows = self.add_thinking_rows(
            x_BRiCLD, single_eval_pos=num_train_labels
        )
        
        num_channels = x_BRCLD.shape[2]
        if num_channels == 1:
            x_BRCLD = x_BRCLD.squeeze(2)
            for block in self.blocks:
                if force_recompute_layer:
                    x_BRCLD = torch.utils.checkpoint.checkpoint(
                        block,
                        x_BRCLD,
                        num_train_and_thinking_rows,
                        save_peak_memory_factor,
                        use_reentrant=False,
                    )
                else:
                    x_BRCLD = block(
                        x_BRCLD, num_train_and_thinking_rows, save_peak_memory_factor
                    )
        else:
            # Loop over channels
            for block, channel_block in zip(self.blocks, self.channel_blocks):
                for c in range(num_channels):
                    x_BRCLD_c = x_BRCLD[:, :, c]
                    if force_recompute_layer:
                        x_BRCLD_c = torch.utils.checkpoint.checkpoint(
                            block,
                            x_BRCLD_c,
                            num_train_and_thinking_rows,
                            save_peak_memory_factor,
                            use_reentrant=False,
                        )
                    else:
                        x_BRCLD_c = block(
                            x_BRCLD_c, num_train_and_thinking_rows, save_peak_memory_factor
                        )
                x_BRCLD[:, :, c] = x_BRCLD_c

                x_BRCD = x_BRCLD[:,:,:,:-1].mean(dim=-2) # Average over time dimension
                if force_recompute_layer:
                    x_BRCD = torch.utils.checkpoint.checkpoint(
                        channel_block,
                        x_BRCD,
                        num_train_and_thinking_rows,
                        save_peak_memory_factor,
                        use_reentrant=False,
                    )
                else:
                    x_BRCD = channel_block(
                        x_BRCD, num_train_and_thinking_rows, save_peak_memory_factor
                    )

                x_BRCLD[:,:,:,:-1] += x_BRCD.unsqueeze(-2) # Add the channel output back to the original tensor

                    
        # M: number of test samples
        # B: batch size
        test_embeddings_BMCD = x_BRCLD[:, num_train_and_thinking_rows:, -1]
        test_embeddings_MBCD = test_embeddings_BMCD.transpose(0, 1)
        
        # N: number of training rows
        train_rows_start = self.add_thinking_rows.num_thinking_rows
        train_rows_end = num_train_and_thinking_rows
        train_embeddings_BNCD = x_BRCLD[:, train_rows_start:train_rows_end, -1]
        train_embeddings_NBCD = train_embeddings_BNCD.transpose(0, 1)

        if task_type == "linear_projection":
            # Average over channels
            test_embeddings_MBD = test_embeddings_MBCD.mean(dim=2)
            test_output_MB1 = self.output_projection(test_embeddings_MBD)
        
        elif task_type == "output_embedding":
            test_output_MB1 = None

        elif task_type == "many_class_decoder":
            # Average over channels
            test_embeddings_MBD = test_embeddings_MBCD.mean(dim=2)
            train_embeddings_NBD = train_embeddings_NBCD.mean(dim=2)
            y_BN = y.transpose(0, 1) if y.dim() == 2 else y.unsqueeze(0)
            test_output_MB1: Tensor = self.many_class_decoder(
                train_embeddings_NBD.transpose(0, 1),
                test_embeddings_MBD.transpose(0, 1),
                y_BN[:, :num_train_and_thinking_rows],
            )

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        if only_return_standard_out:
            return test_output_MB1

        output = {"standard": test_output_MB1.squeeze(1)}
        output["train_embeddings"] = train_embeddings_NBCD.squeeze(1)
        output["test_embeddings"] = test_embeddings_MBCD.squeeze(1)

        return output

    def _preprocess_and_embed_features(
        self,
        x_RiBCT: torch.Tensor,
        num_train_labels: int,
        batch_size: int,
        num_channels: int,
    ) -> torch.Tensor:
        """Preprocess and embed input features and add column embeddings.

        Args:
            x_RiBCT: Input features of shape (Ri, B, C) where:
                - Ri: number of input rows (train + test)
                - B: batch size
                - C: number of columns before grouping
            num_train_labels: Number of training labels for scaling
            batch_size: Batch size for reshaping

        Returns:
            Embedded features of shape (B, Ri, G, X) where:
                - G: number of feature groups
                - X: embedding size
        """
        for channel in range(num_channels):
            x_RiBCT[:, :, :, channel] = _remove_constant_features(x_RiBCT=x_RiBCT[:, :, :, channel])
        # x_RiBCT = _remove_constant_features(x_RiBCT=x_RiBCT)
        # Bg = folded batch size (B * G) and number of feature groups (G)
        x_RiBgCTg, num_feature_groups = _pad_and_reshape_feature_groups(
            x_RiBCT=x_RiBCT,
            num_features_per_group=self.features_per_group,
        )
        nan_and_inf_indicator_RiBgCTg = _generate_nan_and_inf_indicator(x=x_RiBgCTg)
        # For consistency with old base implementation, the imputation is done
        # before the standard scaling
        x_RiBgCTg, _ = _impute_nan_and_inf_with_mean(
            x=x_RiBgCTg, num_train_rows=num_train_labels
        )
        x_RiBgCTg = self.standard_scaler(x=x_RiBgCTg, num_train_rows=num_train_labels)
        x_RiBgCTg = _normalize_feature_groups(
            x_RiBCT=x_RiBgCTg, num_features_per_group=self.features_per_group
        )

        x_RiBgCTg_concat = torch.cat([x_RiBgCTg, nan_and_inf_indicator_RiBgCTg], dim=-1)
        # X = embedding size
        embedded_x_RiBgCX = self.feature_group_embedder(x_RiBgCTg_concat)
        # G = number of feature groups (the number of columns the model will see).
        embedded_x_RiBCGX = embedded_x_RiBgCX.unflatten(
            1, [batch_size, num_feature_groups]
        ).transpose(1, 2)
        embedded_x_BRiCGX = embedded_x_RiBCGX.transpose(0, 1)

        return self._add_column_embeddings(embedded_x_BRiCGX)

    def _preprocess_and_embed_targets(
        self,
        y: torch.Tensor,
        num_train_rows: int,
        num_train_labels: int,
        batch_size: int,
        num_channels: int = 1
    ) -> torch.Tensor:
        """Preprocess and embed the target values.

        Args:
            y: Target values of shape [Rt], [Rt, B], or [Rt, B, 1] where:
                - Rt: number of train rows
                - B: batch size
            num_train_rows: Total number of rows in x
            num_train_labels: Number of training labels for imputation
            batch_size: Batch size for reshaping

        Returns:
            Embedded targets of shape (B, Ri, X) where:
                - Ri: number of input rows (padded)
                - X: embedding size
        """
        y_RiB1 = _prepare_targets(
            y=y,
            num_train_rows=num_train_rows,
            batch_size=batch_size,
        )
        y_nan_and_inf_indicator_RiB1 = _generate_nan_and_inf_indicator(x=y_RiB1)
        y_RiB1 = _impute_target_nan_and_inf(
            y_RiB1=y_RiB1,
            task_type=self.task_type,
            num_train_rows=num_train_labels,
        )

        y_RiB1_concat = torch.cat([y_RiB1, y_nan_and_inf_indicator_RiB1], dim=-1)
        embedded_y_RiBX = self.target_embedder(y_RiB1_concat)
        
        if num_channels > 1:
            # If there are multiple channels, we need to repeat the target embedding for each channel.
            embedded_y_RiBCX = embedded_y_RiBX.repeat(1, 1, num_channels, 1)
        else:
            embedded_y_RiBCX = embedded_y_RiBX.unsqueeze(2)  # Add a channel dimension

        return embedded_y_RiBCX.transpose(0, 1)

    @override
    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Load a state dict, automatically translating base-architecture key names.

        If the state dict uses the old base-architecture naming convention, the
        keys are remapped to the v2.5 names before loading.
        """
        has_base_keys = any(
            k.startswith(("transformer_encoder.", "decoder_dict.", "y_encoder."))
            for k in state_dict
        )
        if has_base_keys:
            state_dict = _replace_keys_from_base_architecture(dict(state_dict))
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def parse_config(config: dict[str, Any]) -> tuple[TabPFNV2p5Config, dict[str, Any]]:
    """Parse the config dict into a TabPFNV2Config, return unused keys.

    Args:
        config: Config dict to parse. This function should use Pydantic to
            verify that it matches the expected schema.

    Returns:
        A tuple, (parsed config, dict containing unused config items).

    Raises:
        pydantic.ValidationError: one or more of the values have the wrong type
    """
    parsed_config = TabPFNV2p5Config(**config)
    return parsed_config, parsed_config.get_unused_config(config)


def get_architecture(
    config: ArchitectureConfig,
    *,
    cache_trainset_representation: bool = False,
) -> TabPFNV2p5:
    """Construct TabPFNV2.5 based on the given config.

    This factory method implements the interface defined in
    tabpfn.architectures.interface.ArchitectureModule.get_architecture().

    Args:
        config: The config returned by parse_config(). This method should use a
            runtime isinstance() check to downcast the config to this architecture's
            specific config class.
        cache_trainset_representation: If True, the model should be configured to
            cache the training data during inference to improve speed.

    Returns: the constructed architecture
    """
    assert isinstance(config, TabPFNV2p5Config)
    if cache_trainset_representation:
        raise NotImplementedError("TabPFNV2.5 does not support kv cache yet.")
    task_type = "multiclass" if config.max_num_classes > 0 else "regression"
    max_n_out = config.max_num_classes if task_type == "multiclass" else config.num_buckets
    return TabPFNV2p5(config=config, max_n_out=max_n_out, task_type=task_type)


def _prepare_targets(
    y: torch.Tensor,
    num_train_rows: int,
    batch_size: int,
) -> torch.Tensor:
    """Pad the targets to the number of rows in x and validate inputs.

    More specifically, we ensure that:
    1. there are test rows (i.e. y_RtB is shorter than num_train_rows),
    2. y is nan-padded to the number of rows in x.
    3. Make sure the target is 3-dimensional.

    Args:
        y: Target values of shape [Rt], [Rt, B], or [Rt, B, 1] where:
            - Rt: number of train rows
            - B: batch size
        num_train_rows: The number of rows in x.
        batch_size: Batch size for reshaping

    Returns:
        A tensor with the shape (Ri, B, 1), where
            - Ri = number of input rows (train + test)
    """
    num_train_labels = y.shape[0]

    # Check that the number of training labels is not greater than
    # the total number of rows.
    # Note: we allow `num_train_labels == num_train_rows` (i.e., no test data) to
    # support use cases like KV-caching and for consistency with the OOM check script
    # (`src/fomo_fitting/scripts/check_oom.py`).
    if num_train_labels > num_train_rows:
        raise ValueError("No test rows provided.")

    # Make sure the target is 3-dimensional.
    target_RBY = y.view(num_train_labels, 1 if y.ndim == 1 else batch_size, -1)
    return torch.nn.functional.pad(
        target_RBY,
        (0, 0, 0, 0, 0, num_train_rows - num_train_labels),
        value=float("nan"),
    )


def _replace_keys_from_base_architecture(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Translate a base-architecture state dict to the v2.5 key names.

    Keys not present in the mapping are passed through unchanged.
    """
    n_layers = sum(k.endswith("self_attn_between_features._w_qkv") for k in state_dict)

    base_to_v2_mapping = [
        ("encoder.5.layer.weight", "feature_group_embedder.weight"),
        ("encoder.5.mlp.0.weight", "feature_group_embedder.0.weight"),
        ("encoder.5.mlp.2.weight", "feature_group_embedder.2.weight"),
        # regression: target linear projection was at position 1
        ("y_encoder.1.layer.weight", "target_embedder.weight"),
        ("y_encoder.1.layer.bias", "target_embedder.bias"),
        # multiclass: target linear projection was at position 2
        ("y_encoder.2.layer.weight", "target_embedder.weight"),
        ("y_encoder.2.layer.bias", "target_embedder.bias"),
        ("decoder_dict.standard.0.weight", "output_projection.0.weight"),
        ("decoder_dict.standard.0.bias", "output_projection.0.bias"),
        ("decoder_dict.standard.2.weight", "output_projection.2.weight"),
        ("decoder_dict.standard.2.bias", "output_projection.2.bias"),
        (
            "feature_positional_embedding_embeddings.weight",
            "feature_positional_embedding_embeddings.weight",
        ),
        (
            "feature_positional_embedding_embeddings.bias",
            "feature_positional_embedding_embeddings.bias",
        ),
        (
            "add_thinking_tokens.row_token_values",
            "add_thinking_rows.row_token_values_TE",
        ),
    ]
    for i in range(n_layers):
        base_to_v2_mapping.extend(
            [
                (
                    f"transformer_encoder.layers.{i}.mlp.linear1.weight",
                    f"blocks.{i}.mlp.0.weight",
                ),
                (
                    f"transformer_encoder.layers.{i}.mlp.linear2.weight",
                    f"blocks.{i}.mlp.2.weight",
                ),
                (
                    f"transformer_encoder.layers.{i}.self_attn_between_features._w_qkv",
                    f"blocks.{i}.per_sample_attention_between_features.qkv_projection.weight",
                ),
                (
                    f"transformer_encoder.layers.{i}.self_attn_between_features._w_out",
                    f"blocks.{i}.per_sample_attention_between_features.out_projection.weight",
                ),
                (
                    f"transformer_encoder.layers.{i}.self_attn_between_items._w_qkv",
                    f"blocks.{i}.per_column_attention_between_cells.qkv_projection.weight",
                ),
                (
                    f"transformer_encoder.layers.{i}.self_attn_between_items._w_out",
                    f"blocks.{i}.per_column_attention_between_cells.out_projection.weight",
                ),
            ]
        )

    new_state_dict: dict[str, torch.Tensor] = {}
    known_base_keys: set[str] = set()
    for base_key, v2_key in base_to_v2_mapping:
        known_base_keys.add(base_key)
        if base_key not in state_dict:
            continue

        # QKV weight has shape (3, num_heads, head_size, input_size).
        # Split into q/k/v weights of shape (num_heads * head_size, input_size).
        if "qkv_projection.weight" in v2_key:
            q_key = v2_key.replace("qkv_projection", "q_projection")
            k_key = v2_key.replace("qkv_projection", "k_projection")
            v_key = v2_key.replace("qkv_projection", "v_projection")
            new_state_dict[q_key] = state_dict[base_key][0].flatten(0, 1)
            new_state_dict[k_key] = state_dict[base_key][1].flatten(0, 1)
            new_state_dict[v_key] = state_dict[base_key][2].flatten(0, 1)
            continue

        # Out-projection weight has shape (num_heads, head_size, output_size).
        # Flatten heads and transpose to match nn.Linear convention (output, input).
        if "out_projection.weight" in v2_key:
            new_state_dict[v2_key] = state_dict[base_key].flatten(0, 1).T
            continue

        new_state_dict[v2_key] = state_dict[base_key]

    for key, value in state_dict.items():
        if key not in known_base_keys:
            new_state_dict[key] = value

    return new_state_dict


def _remove_constant_features(x_RiBCT: torch.Tensor) -> torch.Tensor:
    """Removes constant features from the input data."""
    if x_RiBCT.shape[0] <= 1:
        return x_RiBCT
    print(f"x_RiBCT.shape: {x_RiBCT.shape}")
    column_selection_mask = ~(x_RiBCT[1:] == x_RiBCT[0]).all(0)
    return select_features(x_RiBCT, column_selection_mask.type(torch.bool))


def _pad_and_reshape_feature_groups(
    x_RiBCT: torch.Tensor, num_features_per_group: int
) -> tuple[torch.Tensor, int]:
    """Pad the feature groups and reshape so the feature group dimension is last.

    Returns:
        A tuple of padded and reshaped tensor with shape [Ri, B * G, F] and
        the number of feature groups.
        where:
        - Ri = number of input rows (train + test, before adding thinking rows)
        - B = batch size
        - G = number of feature groups
        - F = number of features per group
    """
    num_columns = x_RiBCT.shape[-1]
    num_padding_features = -num_columns % num_features_per_group
    x_RiBCT_padded = torch.nn.functional.pad(
        x_RiBCT, pad=(0, num_padding_features), value=0
    )
    num_rows, B, num_channels, num_padded_columns = x_RiBCT_padded.shape
    num_feature_groups = num_padded_columns // num_features_per_group

    x_RiBgCTg = x_RiBCT_padded.reshape(
        num_rows, B * num_feature_groups, num_channels, num_features_per_group
    )
    return x_RiBgCTg, num_feature_groups


def _impute_nan_and_inf_with_mean(
    x: torch.Tensor, num_train_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Impute the nan and inf with the mean of the feature.

    Args:
        x: Tensor of shape [R, ...], where
            R = number of rows
        num_train_rows: The position to use for single evaluation.

    Returns:
        A tuple of (imputed tensor, nan/inf mask).
    """
    feature_means = torch_nanmean(x=x[:num_train_rows], axis=0, include_inf=True)
    nan_mask = torch.logical_or(torch.isnan(x), torch.isinf(x))
    return torch.where(nan_mask, feature_means.unsqueeze(0).expand_as(x), x), nan_mask


def _impute_target_nan_and_inf(
    y_RiB1: torch.Tensor,
    task_type: TaskType,
    num_train_rows: int,
) -> torch.Tensor:
    """Impute NaN/Inf values in the target tensor.

    Args:
        y_RiB1: Tensor of shape [Ri, B, 1], where Ri is the number of train+test rows.
        task_type: The task type ("regression" or "multiclass").
        num_train_rows: Number of training rows.

    """
    if task_type == "regression":
        y_RiB1, _ = _impute_nan_and_inf_with_mean(
            x=y_RiB1, num_train_rows=num_train_rows
        )
        return y_RiB1

    # The following class imputation is performed for backwards compatibility.
    # We impute the mean and then do a ceil() operation.
    # Only apply ceil() to imputed positions to preserve differentiability for
    # original values (e.g. during prompt tuning).
    y_RiB1, nan_inf_mask = _impute_nan_and_inf_with_mean(
        x=y_RiB1, num_train_rows=num_train_rows
    )
    return torch.where(nan_inf_mask, y_RiB1.ceil(), y_RiB1)


def _normalize_feature_groups(
    x_RiBCT: torch.Tensor,
    num_features_per_group: int,
) -> torch.Tensor:
    """Normalize the feature groups."""
    Ri = x_RiBCT.shape[0]
    non_constant_mask = (x_RiBCT[1:] == x_RiBCT[0]).sum(0) != (Ri - 1)
    number_of_used_features = torch.clip(
        non_constant_mask.sum(-1).unsqueeze(-1),
        min=1,
    ).to(x_RiBCT.device)
    scale = num_features_per_group / number_of_used_features
    x_RiBCT = x_RiBCT * torch.sqrt(scale)

    return torch.where(
        non_constant_mask.unsqueeze(0).expand_as(x_RiBCT),
        x_RiBCT,
        torch.zeros_like(x_RiBCT),
    )


def _generate_nan_and_inf_indicator(x: torch.Tensor) -> torch.Tensor:
    """Generate the nan and inf indicators for a generic input tensor."""
    return (
        torch.isnan(x) * NAN_INDICATOR
        + torch.logical_and(torch.isinf(x), torch.sign(x) == 1) * INFINITY_INDICATOR
        + torch.logical_and(torch.isinf(x), torch.sign(x) == -1)
        * NEG_INFINITY_INDICATOR
    ).to(x.dtype)

class CubePFNEncoder(nn.Module, ABC):
    def __init__(
        self,
        seed: int,
        tabpfn_kwargs: dict,
        use_checkpoint: bool = True,
        pretraining: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.pretraining = pretraining

        resolved = str(tabpfn_kwargs.pop("model_path"))
        checkpoint = _load_checkpoint_cached(resolved, Checkpoint(resolved).identity())

        full_state = checkpoint["state_dict"]
        criterion_state_keys = [k for k in full_state if "criterion." in k]

        if "performance_options" in tabpfn_kwargs:
            performance_options = tabpfn_kwargs.pop("performance_options")
        else:
            performance_options = {}

        model = TabPFNV2p5(**tabpfn_kwargs, device="cuda", dtype=torch.float32)

        if use_checkpoint:
            model_state = {k: v for k, v in full_state.items() if k not in criterion_state_keys}
            missing_keys = []
            unexpected_keys = []
            for k in model_state.keys():
                if k not in model.state_dict():
                    unexpected_keys.append(k)
            for k in model.state_dict().keys():
                if k not in model_state:
                    missing_keys.append(k)
            if missing_keys:
                _logger.warning("Missing keys in model state dict: %s", missing_keys)
            if unexpected_keys:
                _logger.warning("Unexpected keys in model state dict: %s", unexpected_keys)
            model.load_state_dict(model_state, strict=True)


        self.model = model
        self.performance_options = PerformanceOptions(**performance_options)

    # def forward(self, ts: Tensor, ts_diff: Tensor, ts_ftt: Tensor, ts_croped: Tensor, y: Tensor, **kwargs) -> Tensor:
    def forward(self, ts: Tensor, y: Optional[Tensor] = None, ts_diff: Optional[Tensor] = None, **kwargs) -> Tensor:
        """Forward pass of the TSPFN encoder, using the forward pass of the TabPFN-2p5 model.

        Args:
            ts (Tensor): time series input tensor of shape (Ri, B, C, T)
            y (Tensor): target tensor of shape (Ri, B)
            ts_diff (Tensor): time series difference input tensor of shape (Ri, B, C, T)

        Returns:
            Tensor: Embeddings of shape (Ri, C, E) for each input time series.
        """
        # x is (Ri, B, C, T) and y is (train_size, B)
        # x = x.flatten(-2)  # (Ri, B, C, T) -> (Ri, B, C*T)
        ts_emb = self.model(ts, y, ts_diff, performance_options=self.performance_options).squeeze(0)
        if not self.pretraining:
            return ts_emb["standard"]
        else:
            ts_emb = torch.cat([ts_emb["train_embeddings"], ts_emb["test_embeddings"]], dim=0)
            return ts_emb