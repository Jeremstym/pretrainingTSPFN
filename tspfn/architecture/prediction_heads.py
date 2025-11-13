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

ModuleType = Union[str, Callable[..., nn.Module]]


class PFNPredictionHead(nn.Module):
    """Prediction head architecture described in the TabPFN paper."""

    def __init__(
        self,
        in_features: int,
        model_path: Path,
    ):
        """Initializes class instance.

        Args:
            in_features: Number of features in the input feature vector.
            out_features: Number of features to output.
        """
        super().__init__()
        self.head = load_model_criterion_config(
            model_path=model_path,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            version="v2",
            download=False,
        )[0].decoder_dict["standard"]

    def forward(self, x: Tensor, n_class: int) -> Tensor:
        """Predicts unnormalized features from a feature vector input.

        Args:
            x: (N, `in_features`), Batch of feature vectors.

        Returns:
            - (N, `out_features`), Batch of output features.
        """
        x = self.head(x.unsqueeze(1)).squeeze(1)
        x = x[:, :n_class]  # Original TabPFN prediction head outputs 10 classes by default, reduce to n_class

        return x
