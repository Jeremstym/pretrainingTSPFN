import torch
import einops
from torch import nn
import torch.nn.functional as F
from typing import Tuple
from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention

# def _apply_rope(x, seq_len, inv_freq: torch.Tensor, head_dim: int) -> torch.Tensor:
#         # x shape: [Batch, Heads, Seq_Len, Head_Dim]
#         t = torch.arange(seq_len, device=x.device).type_as(inv_freq)
#         freqs = torch.outer(t, inv_freq)  # [Seq_Len, Head_Dim/2]
#         emb = torch.cat((freqs, freqs), dim=-1) # [Seq_Len, Head_Dim]

#         # Reshape for broadcasting: [1, 1, Seq_Len, Head_Dim]
#         cos = emb.cos()[None, None, :, :]
#         sin = emb.sin()[None, None, :, :]

#         # Rotate half logic
#         x1 = x[..., :head_dim//2]
#         x2 = x[..., head_dim//2:]
#         x_rotated = torch.cat((-x2, x1), dim=-1)

#         return (x * cos) + (x_rotated * sin)


# 1. Define the rotation logic
def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x, d_k):
    # x shape: [batch, seq_len, nhead, d_k]
    n = x.shape[1]
    device = x.device

    # Standard RoPE frequencies
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float().to(device) / d_k))
    t = torch.arange(n, device=device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    # [1, seq_len, 1, d_k] for broadcasting
    cos, sin = emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]
    return (x * cos) + (_rotate_half(x) * sin)


# 2. Define the new logic
def rope_compute_heads_wrapper(q, k, v, kv, qkv, dropout_p=None, softmax_scale=None):
    # Step A: Let the existing static logic unbind the tensors
    # We use the class name to call the original static method
    if qkv is not None:
        q, k, v = qkv.unbind(dim=-3)
    elif kv is not None:
        k, v = kv.unbind(dim=-3)

    # Step B: Apply RoPE to the unbundled Q and K
    d_k = q.shape[-1]
    q = _apply_rope(q, d_k)
    k = _apply_rope(k, d_k)

    # Step C: Call original static method with our rotated tensors
    # We set qkv and kv to None so the original method uses our provided q, k, v
    return MultiHeadAttention.compute_attention_heads(
        q=q, k=k, v=v, kv=None, qkv=None, dropout_p=dropout_p, softmax_scale=softmax_scale
    )


def interpolate_pos_encoding(pos_embed, new_len):
    # Current shape: [1, 1, old_len, embed_dim]
    old_len = pos_embed.shape[2]
    embed_dim = pos_embed.shape[3]
    
    if old_len == new_len:
        return pos_embed

    # Then permute to [Batch, Channels, Length] -> [1, embed_dim, old_len]
    x = pos_embed.squeeze(0).permute(0, 2, 1)

    # 'linear' is the standard for 1D. 
    # 'bicubic' is only for 2D inputs (Height x Width).
    x = F.interpolate(
        x, 
        size=new_len, 
        mode="linear", 
        align_corners=False
    )

    # then unsqueeze to get back to [1, 1, new_len, embed_dim]
    x = x.permute(0, 2, 1).unsqueeze(0)
    
    return x
