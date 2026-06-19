import math
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
from pathlib import Path

import torch
from scipy.special import binom, factorial
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init

from tabpfn.model_loading import load_model_criterion_config

import logging

ModuleType = Union[str, Callable[..., nn.Module]]


class LinearContrastiveHead(nn.Module):
    """Prediction head architecture described in the TabPFN paper."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            in_features: Number of features in the input feature vector.
            out_features: Number of features to output.
        """
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, out_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Predicts unnormalized features from a feature vector input.

        Args:
            x: (S, `in_features`=E), Batch of feature vectors.

        Returns:
            - (S, `out_features`), Batch of output features.
        """
        x_proj = self.projection_head(x)

        return x_proj
