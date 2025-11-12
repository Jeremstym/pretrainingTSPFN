import math
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

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
        self, in_features: int, out_features: int, use_layernorm: bool = False, model_path: Optional[str] = None
    ):
        """Initializes class instance.

        Args:
            in_features: Number of features in the input feature vector.
            out_features: Number of features to output.
        """
        super().__init__()
        self.model_path = model_path
        self.n_class = out_features

        if self.model_path is None:
            self.module_dict = nn.ModuleDict()
            if use_layernorm:
                self.module_dict["layernorm"] = nn.LayerNorm(in_features)
            self.module_dict["fc1"] = nn.Linear(in_features, 4 * in_features)
            self.module_dict["gelu"] = nn.GELU()
            self.module_dict["fc2"] = nn.Linear(4 * in_features, out_features)

            self.head = nn.Sequential(*self.module_dict.values())

        else:
            self.head = load_model_criterion_config(
                model_path=model_path,
                check_bar_distribution_criterion=False,
                cache_trainset_representation=False,
                which="classifier",
                version="v2",
                download=False,
            )[0].decoder_dict["standard"]

    def forward(self, x: Tensor) -> Tensor:
        """Predicts unnormalized features from a feature vector input.

        Args:
            x: (N, `in_features`), Batch of feature vectors.

        Returns:
            - (N, `out_features`), Batch of output features.
        """
        if self.model_path is None:
            x = self.head(x)
        else:
            x = self.head(x.unsqueeze(1)).squeeze(1)
            x = x[:, : self.n_class]

        return x
