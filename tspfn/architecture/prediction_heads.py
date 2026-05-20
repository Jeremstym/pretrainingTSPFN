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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ModuleType = Union[str, Callable[..., nn.Module]]


class PFNPredictionHead(nn.Module):
    """Prediction head architecture described in the TabPFN paper."""

    def __init__(
        self,
        in_features: int,
        tabpfn_kwargs: dict,
        num_classes: int = 10, # Default to 10 classes as in original TabPFN
        updated_pfn_path: Union[Path, None] = None,
        random_init: bool = False,
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
        
        if random_init:
            model.apply(self._init_weights)
            logger.info("Randomly initialized TabPFN model weights")
        else:
            logger.info("Loaded pretrained TabPFN model weights")

        if updated_pfn_path is not None:
            # Load updated model weights after pretraining
            state_dict = torch.load(updated_pfn_path, map_location="cuda:0")
            # new_state_dict = {}
            # for k, v in state_dict.items():
            #     if k.startswith("model."):
            #         new_key = k[len("model.") :]  # strip the prefix
            #         new_state_dict[new_key] = v
            #     else:
            #         new_state_dict[k] = v
            if "pe" in state_dict:
                state_dict.pop("pe")  # Remove pe if present
            if "channel_positional_encoding" in state_dict:
                state_dict.pop("channel_positional_encoding")  # Remove channel_positional_encoding if present
            if "cwpe.weight" in state_dict:
                state_dict.pop("cwpe.weight")  # Remove cwpe weights if present
            model.load_state_dict(state_dict, strict=True)

        self.head = model.decoder_dict["standard"]

    def _init_weights(self, module):
        # Initialize Linear layers (with or without bias)
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Initialize LayerNorm layers (weight to 1, bias to 0)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Initialize embeddings using normal distribution if any
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=1)
    
    def forward(self, x: Tensor, num_classes: Optional[int] = None) -> Tensor:
        """Predicts unnormalized features from a feature vector input.

        Args:
            x: (N, `in_features`), Batch of feature vectors.
            num_classes: Number of classes for prediction.

        Returns:
            - (N, `out_features`), Batch of output features.
        """
        if num_classes is not None:
            self.n_classes = num_classes
        x = self.head(x.unsqueeze(1)).squeeze(1)
        # x = x[:, :n_class]  # Original TabPFN prediction head outputs 10 classes by default, reduce to n_class
        x = x[:, :, :self.n_classes]  # Original TabPFN prediction head outputs 10 classes by default, reduce to n_class

        return x

class PredictionHead(nn.Module):
    """Prediction head architecture described in the TabPFN paper."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = 10, # Default to 10 classes as in original TabPFN
    ):
        """Initializes class instance.

        Args:
            in_features: Number of features in the input feature vector.
            out_features: Number of features to output.
        """
        super().__init__()
        self.n_classes = num_classes
        self.head = nn.Linear(in_features, num_classes)    
    def forward(self, x: Tensor) -> Tensor:
        """Predicts unnormalized features from a feature vector input.

        Args:
            x: (N, `in_features`), Batch of feature vectors.

        Returns:
            - (N, `out_features`), Batch of output features.
        """
        x = self.head(x)
        return x
