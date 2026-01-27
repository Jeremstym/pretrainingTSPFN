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
        tabpfn_kwargs: dict,
        num_classes: int,
        updated_pfn_path: Union[Path, None] = None,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            in_features: Number of features in the input feature vector.
            out_features: Number of features to output.
        """
        super().__init__()
        list_model, _, _, _ = load_model_criterion_config(**tabpfn_kwargs)
        model = list_model[0]
        self.n_classes = num_classes
        
        # TODO: for now, we only support the standard prediction head
        # if updated_pfn_path is not None:
        #     # Load updated model weights after pretraining
        #     state_dict = torch.load(updated_pfn_path, map_location="cuda:0") # updated_pfn_path is already a state dict
        #     new_state_dict = {}
        #     for k, v in state_dict.items():
        #         if k.startswith("model."):
        #             new_key = k[len("model."):]  # strip the prefix
        #             new_state_dict[new_key] = v
        #         else:
        #             new_state_dict[k] = v
        #     model.load_state_dict(new_state_dict, strict=True)

        self.head = model.decoder_dict["standard"]
    
    def forward(self, x: Tensor) -> Tensor:
        """Predicts unnormalized features from a feature vector input.

        Args:
            x: (N, `in_features`), Batch of feature vectors.

        Returns:
            - (N, `out_features`), Batch of output features.
        """
        x = self.head(x.unsqueeze(1)).squeeze(1)
        # x = x[:, :n_class]  # Original TabPFN prediction head outputs 10 classes by default, reduce to n_class
        x = x[:, :, :self.n_classes]  # Original TabPFN prediction head outputs 10 classes by default, reduce to n_class

        return x
