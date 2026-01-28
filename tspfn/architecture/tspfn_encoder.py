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
        seed: int,
        tabpfn_kwargs: dict,
        features_per_group: int,
        updated_pfn_path: Union[Path, None] = None,
        random_init: bool = False,
        recompute_layer: bool = True,
        **kwargs,
    ):
        super().__init__()
        list_model, _, self.model_config, _ = load_model_criterion_config(**tabpfn_kwargs)
        self.model = list_model[0]
        if updated_pfn_path is not None:
            # Load updated model weights after pretraining
            logging.info(f"Loading updated TabPFN model weights from {updated_pfn_path}")
            state_dict = torch.load(updated_pfn_path, map_location="cuda:0")  # updated_pfn_path is already a state dict
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_key = k[len("model.") :]  # strip the prefix
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict, strict=True)

        self.encoder = self.model.encoder
        self.y_encoder = self.model.y_encoder
        self.transformer_encoder = self.model.transformer_encoder
        self.features_per_group = features_per_group  # 1 for TabPFN v2, 3 for TabPFN v2.5
        self.recompute_layer = recompute_layer

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

    def sinusoidal_positional_encoding(self, sequence_length=499, embedding_dim=192, n=10000.0):
        if embedding_dim % 2 != 0:
            raise ValueError("Embedding dimension must be even")

        positions = torch.arange(0, sequence_length).unsqueeze(1)  # Shape: (sequence_length, 1)
        denominators = torch.pow(
            n, 2 * torch.arange(0, embedding_dim // 2) / embedding_dim
        )  # Shape: (embedding_dim/2,)

        posenc = torch.zeros(sequence_length, embedding_dim)
        posenc[:, 0::2] = torch.sin(positions / denominators)  # Apply sin to even indices
        posenc[:, 1::2] = torch.cos(positions / denominators)  # Apply cos to odd indices

        return posenc

    def encode_x_and_y(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        single_eval_pos_ = y.shape[0]
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if y.ndim == 2:
            y = y.unsqueeze(-1)  # (Seq, B) -> (Seq, B, 1)

        y = y.transpose(0, 1)  # (B, Seq, 1)

        X = einops.rearrange(X, "s b (f n) -> b s f n", n=self.features_per_group)
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

        y = y.transpose(0, 1)  # (Seq, B, 1)
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
        self, X_full: torch.Tensor, y_train: torch.Tensor, ts_pe: str = "none", *args, **kwargs
    ) -> Tuple[torch.torch.Tensor, torch.torch.Tensor]:

        seq_len, batch_size, num_features = X_full.shape
        emb_x, emb_y, single_eval_pos = self.encode_x_and_y(X_full, y_train)

        if ts_pe == "sinusoidal":
            # Add sinusoidal positional encodings to time series attributes
            pos = self.sinusoidal_positional_encoding().to(emb_x.device)  # (T, E)
            # Broadcast to (B, Seq, T, E)
            pos_broadcasted = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
            emb_x += pos_broadcasted
        elif ts_pe == "none":
            # Use PE from TabPFN model
            emb_x, emb_y = self.model.add_embeddings(
                emb_x,
                emb_y,
                data_dags=None,
                num_features=num_features,
                seq_len=seq_len,
            )
        elif ts_pe == "mixed":
            # Use PE from TabPFN model and add sinusoidal positional encodings to time series attributes
            pos = self.sinusoidal_positional_encoding().to(emb_x.device)  # (T, E)
            # Broadcast to (B, Seq, T, E)
            pos_broadcasted = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
            emb_x, emb_y = self.model.add_embeddings(
                emb_x,
                emb_y,
                data_dags=None,
                num_features=num_features,
                seq_len=seq_len,
            )
            emb_x += pos_broadcasted
        else:
            raise ValueError(f"Unknown ts_pe option: {ts_pe}")

        # (B, Seq, num_features, d_model) + (B, Seq, 1, d_model) -> (B, Seq, num_features + 1, d_model)
        embedded_input = torch.cat((emb_x, emb_y.unsqueeze(2)), dim=2)
        assert not torch.isnan(
            embedded_input
        ).any(), f"{torch.isnan(embedded_input).any()=}, Make sure to add nan handlers"

        output = self.transformer_encoder(
            embedded_input,
            single_eval_pos=single_eval_pos,
            cache_trainset_representation=False,
            recompute_layer=self.recompute_layer,
            save_peak_mem_factor=None,
        )
        out_query = output[:, single_eval_pos:, :] # (B, Query, num_features + 1, d_model)

        return out_query