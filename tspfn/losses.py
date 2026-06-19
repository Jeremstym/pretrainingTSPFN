from torch import Tensor
from typing import Union, Callable, Dict, List, Literal, Optional, Tuple
import torch
from torch import nn

# class ContrastiveAugmentationLoss(nn.Module):
#     """
#     A contrastive loss used for pre-training.

#     Parameters
#     ----------
#     temperature: float, default=0.1
#         Temperature scaling parameter used to regulate the sharpness of the softmax operator.
#     device: {'cpu', 'cuda'}, default='cuda'
#         On which device the model is located.
#     """

#     def __init__(self, temperature=0.1, device="cuda"):
#         super().__init__()
#         self.temperature = temperature
#         self.device = device

#     def forward(self, q, k):
#         q = nn.functional.normalize(q, dim=1)
#         k = nn.functional.normalize(k, dim=1)
#         logits = torch.einsum("nc,ck->nk", [q, k.t()])
#         logits /= self.temperature
#         labels = torch.arange(q.shape[0], dtype=torch.long).to(self.device)
#         return nn.CrossEntropyLoss()(logits, labels)


class ContrastiveAugmentationLoss(nn.Module):
    """
    A contrastive loss used for pre-training.

    Parameters
    ----------
    temperature: float, default=0.1
        Temperature scaling parameter used to regulate the sharpness of the softmax operator.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def _compute_loss(self, q, k):
        # q = nn.functional.normalize(q, dim=1)
        # k = nn.functional.normalize(k, dim=1)
        logits = torch.einsum("nc,ck->nk", [q, k.t()])
        logits /= self.temperature
        labels = torch.arange(q.shape[0], dtype=torch.long).to(q.device)
        return nn.CrossEntropyLoss()(logits, labels)

    def forward(self, emb_proj):
        """Performs a forward pass through the loss function.

        Args:
            emb_proj: (S, C, E), Projected embeddings with channels.
        """
        num_channels = emb_proj.shape[1]
        loss_val = 0
        for channel in range(num_channels):
            for second_channel in range(num_channels):
                if channel != second_channel:
                    loss_val += self._compute_loss(emb_proj[:, channel, :], emb_proj[:, second_channel, :])
        return loss_val / (num_channels * (num_channels - 1))


class ContrastiveChannelLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy Loss with Decoupling."""

    def __init__(self, temperature: float = 0.1):
        """Initializes class instance.

        Args:
            temperature: Temperature scaling factor.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, ts_proj: Tensor, diff_proj: Tensor, freq_proj: Tensor, crop_proj: Tensor) -> Tensor:
        """Performs a forward pass through the loss function.

        Args:
            ts_proj: (S, E), Projected embeddings.
            diff_proj: (S, E), Projected embeddings.
            freq_proj: (S, E), Projected embeddings.
            crop_proj: (S, E), Projected embeddings.

        Returns:
            Scalar loss value.
        """

        logits_ts_diff = torch.einsum("nc,ck->nk", [ts_proj, diff_proj.t()]) / self.temperature
        logits_ts_freq = torch.einsum("nc,ck->nk", [ts_proj, freq_proj.t()]) / self.temperature
        logits_ts_crop = torch.einsum("nc,ck->nk", [ts_proj, crop_proj.t()]) / self.temperature

        labels = torch.arange(ts_proj.shape[0], dtype=torch.long).to(ts_proj.device)

        loss1 = nn.CrossEntropyLoss()(logits_ts_diff, labels)
        loss2 = nn.CrossEntropyLoss()(logits_ts_freq, labels)
        loss3 = nn.CrossEntropyLoss()(logits_ts_crop, labels)

        return (loss1.mean() + loss2.mean() + loss3.mean()) / 3
