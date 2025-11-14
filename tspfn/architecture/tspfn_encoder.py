import os
from typing import Tuple, Union
from shutil import copy2
from pathlib import Path

from abc import ABC
import einops
import torch
import torch.nn as nn
from tabpfn.model_loading import load_model_criterion_config

import logging


class TSPFNEncoder(nn.Module, ABC):
    def __init__(
        self,
        model_path: Path,
        which: str,
        fit_mode: str,
        seed: int,
        updated_pfn_path: Union[Path, None] = None,
        random_init: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.model, _, self.model_config = load_model_criterion_config(
            model_path=model_path,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=(fit_mode == "fit_with_cache"),
            which="classifier",
            version="v2",
            download=False,
        )
        if updated_pfn_path is not None:
            # Load updated model weights after pretraining
            logging.info(f"Loading updated TabPFN model weights from {updated_pfn_path}")
            state_dict = torch.load(updated_pfn_path, map_location="cuda:0") # updated_pfn_path is already a state dict
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_key = k[len("model."):]  # strip the prefix
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict, strict=True)

        self.encoder = self.model.encoder
        self.y_encoder = self.model.y_encoder
        self.transformer_encoder = self.model.transformer_encoder
        self.model.features_per_group = 1  # Each feature is its own group

        if random_init:  # random_init:
            self.model.apply(self._init_weights)
            logging.info("Randomly initialized TabPFN model weights")
        else:
            logging.info("Loaded pretrained TabPFN model weights")

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

    def encode_x_and_y(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        single_eval_pos_ = y.shape[0]
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if y.ndim == 2:
            y = y.unsqueeze(-1)  # (Seq, N) -> (Seq, N, 1)

        y = y.transpose(0, 1)  # (N, Seq, 1)

        assert y.shape[1] == single_eval_pos_

        X = einops.rearrange(X, "s b (f n) -> b s f n", n=self.model.features_per_group)
        y = torch.cat(
            (
                y,
                torch.nan
                * torch.zeros(
                    y.shape[0],
                    X.shape[1] - y.shape[1],
                    y.shape[2],
                    device=y.device,
                    dtype=y.dtype,
                ),
            ),
            dim=1,
        )

        y = y.transpose(0, 1)  # (Seq, N, 1)
        y[single_eval_pos_:] = torch.nan  # Make sure that no label leakage ever happens

        embedded_y = self.y_encoder(
            y,
            single_eval_pos=single_eval_pos_,
            cache_trainset_representation=False,
        ).transpose(0, 1)

        assert not torch.isnan(embedded_y).any(), f"{torch.isnan(embedded_y).any()=}, Make sure to add nan handlers"

        X = einops.rearrange(X, "b s f n -> s (b f) n")
        embedded_x = einops.rearrange(
            self.encoder(
                {"main": X},
                single_eval_pos=single_eval_pos_,
                cache_trainset_representation=False,
            ),
            "s (b f) e -> b s f e",
            b=embedded_y.shape[0],
        )
        return embedded_x, embedded_y, single_eval_pos_

    def forward(
        self, X_full: torch.Tensor, y_train: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.torch.Tensor, torch.torch.Tensor]:

        seq_len, batch_size, num_features = X_full.shape

        emb_x, emb_y, single_eval_pos = self.encode_x_and_y(X_full, y_train)
        emb_x, emb_y = self.model.add_embeddings(
            emb_x,
            emb_y,
            data_dags=None,
            num_features=num_features,
            seq_len=seq_len,
        )

        # (N, Seq, num_features, d_model) + (N, Seq, 1, d_model) -> (N, Seq, num_features + 1, d_model)
        embedded_input = torch.cat((emb_x, emb_y.unsqueeze(2)), dim=2)
        assert not torch.isnan(
            embedded_input
        ).any(), f"{torch.isnan(embedded_input).any()=}, Make sure to add nan handlers"

        output = self.transformer_encoder(
            embedded_input,
            single_eval_pos=single_eval_pos,
            cache_trainset_representation=False,
        )
        out_query = output[:, single_eval_pos:, :].transpose(0, 1)
        query_encoder_out = out_query.squeeze(1)  # (N_query, S_tab, d_model)

        return query_encoder_out
