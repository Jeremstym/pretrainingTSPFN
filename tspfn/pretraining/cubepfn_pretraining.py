import os
import csv
import numpy as np
import logging
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, cast, Union

import hydra
from omegaconf import DictConfig
import torch
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torch.utils.data import DataLoader
from torch import Tensor, nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryAveragePrecision,
    BinaryCohenKappa,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassAveragePrecision,
    MulticlassCohenKappa,
    MulticlassRecall,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics import MetricCollection

from data.utils.decorators import auto_move_data
from tspfn.system import TSPFNSystem
from tspfn.utils import (
    half_batch_split,
    stratified_batch_split,
    z_scoring,
    z_scoring_per_channel,
    get_stratified_batch_split,
)

logger = logging.getLogger(__name__)


class CubePFNPretraining(TSPFNSystem):
    """Multi-modal transformer to learn a representation from cardiac imaging and patient records data."""

    def __init__(
        self,
        embed_dim: int,
        split_finetuning: float = 0.5,
        contrastive_losses: Optional[Dict[str, Callable[[Tensor, Tensor], Tensor]] | DictConfig] = None,
        # time_series_positional_encoding: Literal["none", "sinusoidal", "learned"] = "none",
        *args,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            embed_dim: Size of the tokens/embedding for all the modalities.
            contrastive_losses: Contrastive criteria to measure the similarity between different views of the same sample.
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        # Ensure string tags are converted to their appropriate enum types
        # And do it before call to the parent's `init` so that the converted values are saved in `hparams`

        super().__init__(*args, **kwargs)

        self.embed_dim = embed_dim

        # Add shortcut to lr to work with Lightning's learning rate finder
        # self.hparams["lr"] = None

        # Contrastive losses and metrics
        self.contrastive_losses = {}
        if contrastive_losses is not None:
            self.contrastive_losses = {
                contrastive_task: (
                    hydra.utils.instantiate(target_loss) if isinstance(target_loss, DictConfig) else target_loss
                )
                for contrastive_task, target_loss in contrastive_losses.items()
            }

        # Initialize transformer encoder and self-supervised + contrastive heads
        self.encoder, self.contrastive_head = self.configure_model()

        # scales each time-series w.r.t. its mean and std
        self.ts_scaler = lambda x: (x - torch.mean(x, axis=-1, keepdim=True)) / (
            torch.std(x, axis=-1, keepdim=True) + 1e-5
        )
        self.crop_resize1 = hydra.utils.instantiate(
            self.hparams["crop_resize"],
        )
        self.crop_resize2 = hydra.utils.instantiate(
            self.hparams["crop_resize"],
        )
        self.differentiate = hydra.utils.instantiate(
            self.hparams["differentiate"],
        )

        # self.time_series_positional_encoding = time_series_positional_encoding

    @property
    def example_input_array(self) -> Tensor:
        """Redefine example input array based on the cardiac attributes provided to the model."""
        # 2 is the size of the batch in the example
        time_series_attrs = torch.randn(1, 10, 4, 512)  # (B=1, S, C, T)
        return time_series_attrs  # (B=1, S, C, T)

    def configure_model(
        self,
    ) -> Tuple[nn.Module, nn.Module, Optional[nn.ModuleDict]]:
        """Build the model, which must return a transformer encoder, and self-supervised or contrastive heads."""
        # Build the transformer encoder
        encoder = hydra.utils.instantiate(self.hparams["model"]["encoder"])

        # Build the contrastive heads following the architecture proposed in
        # https://arxiv.org/pdf/2106.11959
        contrastive_head = None
        if self.contrastive_losses is not None and len(self.contrastive_losses) > 0:
            contrastive_head = hydra.utils.instantiate(
                self.hparams["model"]["contrastive_head"],
                in_features=self.embed_dim,
                out_features=self.embed_dim,
            )
        return encoder, contrastive_head

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure optimizer to ignore parameters that should remain frozen (e.g. tokenizers)."""
        # Frozen tabpfn encoder
        if self.hparams["model"].get("freeze_encoder", False):
            for param in self.encoder.parameters():
                param.requires_grad = False

        optimizer_grouped_parameters = None
        return super().configure_optimizers(params=optimizer_grouped_parameters)

    @auto_move_data
    def encode(
        self,
        time_series_attrs: Tensor,
    ) -> Tensor:
        """Embeds input sequences using the encoder model, optionally selecting/pooling output ts for the embedding.

        Args:
            ts: (S, C, T), Tokens to feed to the encoder.
        Returns:
            (S, C, E), Embedded features for each token in the input sequence.
        """
        time_series_attrs = time_series_attrs.squeeze(0)  # (S, C, T)
        if time_series_attrs.ndim == 2:
            time_series_attrs = time_series_attrs.unsqueeze(1)  # (S, T) -> (S, C=1, T)
            ts = self.ts_scaler(time_series_attrs).unsqueeze(1)  # (S, B, C, T)
        ts = self.ts_scaler(time_series_attrs).unsqueeze(1)  # (S, B, C, T)
        # ts_diff = self.ts_scaler(self.differentiate(time_series_attrs))  # (S, C, T-1)
        # ts_diff = F.pad(ts_diff, (0, 1), mode="constant", value=0).unsqueeze(1)  # Match original length (S, B, C, T)
        ts_augmented1 = self.crop_resize1(time_series_attrs).unsqueeze(1)  # (S, B, C, T)
        # ts_diff_aug1 = self.differentiate(ts_augmented1.squeeze(1)).unsqueeze(1)  # (S, B, C, T-1)
        # ts_diff_aug1 = F.pad(ts_diff_aug1, (0, 1), mode="constant", value=0)  # Match original length (S, B, C, T)
        ts_augmented2 = self.crop_resize2(time_series_attrs).unsqueeze(1)  # (S, B, C, T)
        # ts_diff_aug2 = self.differentiate(ts_augmented2.squeeze(1)).unsqueeze(1)  # (S, B, C, T-1)
        # ts_diff_aug2 = F.pad(ts_diff_aug2, (0, 1), mode="constant", value=0)  # Match original length (S, B, C, T)

        y_nan = torch.empty(ts.shape[0], 1)  # (S, 1)
        out_aug1 = self.encoder(
            ts=ts_augmented1,
            ts_diff=None,
            y=y_nan,
        )

        out_aug2 = self.encoder(
            ts=ts_augmented2,
            ts_diff=None,
            y=y_nan,
        )

        return out_aug1, out_aug2

    @auto_move_data
    def forward(
        self,
        time_series_attrs: Tensor,
    ) -> Tensor | Dict[str, Tensor]:
        """Performs a forward pass through i) the tokenizer, ii) the transformer encoder and iii) the contrastive head.

        Args:
            time_series_attrs: (S, C, T): time series inputs.
            task: Flag indicating which type of inference task to perform.

        Returns: (S, C, E), Embedded features for each token in the input sequence.
        """
        out_aug1, out_aug2 = self.encode(time_series_attrs)  # (S, C, E)
        ts_proj_aug1 = self.contrastive_head(out_aug1)
        ts_proj_aug2 = self.contrastive_head(out_aug2)

        return ts_proj_aug1, ts_proj_aug2

    def _shared_step(self, batch: Union[Tensor, Tuple[Tensor, ...]], batch_idx: int) -> Dict[str, Tensor]:
        # Shared step for training, validation and testing
        metrics = {}
        losses = []
        assert (
            len(self.contrastive_losses) > 0
        ), "Model must include at least one contrastive loss to perform pretraining."
        metrics.update(self._contrastive_shared_step(batch))
        losses.append(metrics["s_loss"])

        # Compute the sum of the (weighted) losses
        metrics["loss"] = sum(losses)

        return metrics

    def _contrastive_shared_step(
        self,
        batch: Union[Tensor, Tuple[Tensor, ...]],
    ) -> Dict[str, Tensor]:
        # Forward pass through the encoder without gradient computation to fine-tune only the contrastive heads
        assert (
            self.contrastive_head is not None
        ), "You requested to perform a contrastive task, but the model does not include any contrastive heads."

        out_aug1, out_aug2 = self.encode(time_series_attrs=batch)
        ts_proj_aug1 = self.contrastive_head(out_aug1)
        ts_proj_aug2 = self.contrastive_head(out_aug2)

        num_channels = ts_proj_aug1.shape[1]

        # Compute the loss/metrics for each target label, ignoring items for which targets are missing
        losses, metrics = {}, {}

        for contrastive_task, target_loss in self.contrastive_losses.items():
            # if "channel_wise" in contrastive_task:
            #     loss_val = 0
            #     num_channels = ts_proj_aug1.shape[1]
            #     for channel in range(num_channels):
            #         loss_val += target_loss(
            #             ts_proj_aug1[:, channel, :],
            #             ts_proj_aug2[:, channel, :],
            #             crop_proj[:, channel, :],
            #         )
            #     metrics.update({f"{contrastive_task}_loss_channel": loss_val})
            # elif "augmentation_wise" in contrastive_task:
            #     loss_val = (
            #         target_loss(ts_proj_aug1) + target_loss(ts_proj_aug2) + target_loss(crop_proj)
            #     )
            #     metrics.update({f"{contrastive_task}_loss_augmentation": loss_val})
            
            if "contrastive_mantis" in contrastive_task:
                loss_val = target_loss(ts_proj_aug1, ts_proj_aug2)
                metrics.update({f"{contrastive_task}_loss_mantis": loss_val})
            else:
                raise ValueError(f"Unknown contrastive task: {contrastive_task}")

            loss_name = f"{target_loss.__class__.__name__.lower().replace('loss', '')}/{contrastive_task}"
            losses[loss_name] = loss_val
        
        losses["s_loss"] = torch.stack(list(losses.values())).mean()
        metrics.update(losses)

        return metrics
