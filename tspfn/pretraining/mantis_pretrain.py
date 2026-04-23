import os
import csv
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
from tspfn.augmentation import RandomCropResize

logger = logging.getLogger(__name__)


class MantisPretraining(TSPFNSystem):
    """Multi-modal transformer to learn a representation from cardiac imaging and patient records data."""

    def __init__(
        self,
        embed_dim: int,
        contrastive_losses: Optional[Dict[str, Callable[[Tensor, Tensor], Tensor]] | DictConfig] = None,
        chunk_size: int = 512,
        *args,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            embed_dim: Size of the tokens/embedding for all the modalities.
            contrastive_losses: Supervised criteria to measure the error between the predicted attributes and their real
                value.
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        # Ensure string tags are converted to their appropriate enum types
        # And do it before call to the parent's `init` so that the converted values are saved in `hparams`

        super().__init__(*args, **kwargs)

        self.chunk_size = chunk_size

        # Add shortcut to lr to work with Lightning's learning rate finder
        self.hparams["lr"] = None

        # Supervised losses and metrics
        self.contrastive_losses = {}
        if contrastive_losses:
            self.contrastive_losses = {
                target_task: (
                    hydra.utils.instantiate(target_loss) if isinstance(target_loss, DictConfig) else target_loss
                )
                for target_task, target_loss in contrastive_losses.items()
            }

        # Initialize transformer encoder and self-supervised + prediction heads
        self.encoder = self.configure_model()

        # # Use ModuleDict so metrics move to GPU automatically
        # metrics_template = MetricCollection(
        #     [
        #         MulticlassAccuracy(num_classes=self.num_classes, average="micro"),
        #         MulticlassAUROC(num_classes=self.num_classes, average="macro"),
        #         MulticlassAveragePrecision(num_classes=self.num_classes, average="macro"),
        #         MulticlassF1Score(num_classes=self.num_classes, average="macro"),
        #         MulticlassCohenKappa(num_classes=self.num_classes),
        #         MulticlassRecall(num_classes=self.num_classes, average="macro"),
        #     ]
        # )
        # # Store them in a dict of ModuleDicts
        # self.metrics = nn.ModuleDict(
        #     {
        #         "train_metrics": nn.ModuleDict({t: metrics_template.clone(prefix="train/") for t in contrastive_losses}),
        #         "val_metrics": nn.ModuleDict({t: metrics_template.clone(prefix="val/") for t in contrastive_losses}),
        #         "test_metrics": nn.ModuleDict({t: metrics_template.clone(prefix="test/") for t in contrastive_losses}),
        #     }
        # )

    @property
    def example_input_array(self) -> Tensor:
        """Redefine example input array based on the cardiac attributes provided to the model."""
        # 2 is the size of the batch in the example
        num_classes = 10  # Default number of classes in TabPFN prediction head
        # labels = torch.cat([torch.randperm(5)]*2)
        labels = torch.arange(10) % num_classes
        labels = labels.unsqueeze(0)
        time_series_attrs = torch.randn(1, 10, 2, 250)  # (B, S, C, T)
        # ts_example_input = torch.cat([time_series_attrs, labels.unsqueeze(-1)], dim=2)  # (B, S, T+1)
        # num_classes = len(torch.unique(labels))
        return time_series_attrs, labels  # (B, S, C, T), (B, S, 1)

    def configure_model(
        self,
    ) -> Tuple[nn.Module, nn.Module, Optional[nn.ModuleDict]]:
        """Build the model, which must return a transformer encoder, and self-supervised or prediction heads."""
        # Build the transformer encoder
        encoder = hydra.utils.instantiate(self.hparams["model"]["encoder"])
        return encoder

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure optimizer to ignore parameters that should remain frozen (e.g. tokenizers)."""
        return super().configure_optimizers()

    @auto_move_data
    def encode(
        self,
        ts_input: Tensor,
        y_input: Optional[Tensor] = None,
    ) -> Tensor:
        """Embeds input sequences using the encoder model, optionally selecting/pooling output ts for the embedding.

        Args:
            y_batch_support: (S, 1), Support set labels.
            ts: (B, S (=Support+Query), C, T), Tokens to feed to the encoder.
        Returns: (B, Query, E), Embeddings of the input sequences.
        """

        # Extract CLS token for constrastive pretraining
        out_features = self.encoder(ts_input)  # (B, E)
    
        return out_features  # (B, E)

    @auto_move_data
    def forward(
        self,
        time_series_attrs: Tensor,
        labels: Tensor,
        task: Literal["encode"] = "encode",
    ) -> Tensor | Dict[str, Tensor]:
        """Performs a forward pass through i) the tokenizer, ii) the transformer encoder and iii) the prediction head.

        Args:
            time_series_attrs: (B, S, T): time series inputs.
            task: Flag indicating which type of inference task to perform.

        Returns:
            if `task` == 'encode':
                (B, Query, E), Batch of features extracted by the encoder.
            if `task` == 'predict' (and the model includes prediction heads):
                1 * (B, Query), Prediction for each target in `losses`.
        """

        # if hasattr(self, "example_input_array") and torch.equal(
        #     time_series_attrs, self.example_input_array[0].to(time_series_attrs.device)
        # ):
        #     summary_mode = True
        # else:
        #     summary_mode = False


        out_features = self.encode(time_series_attrs)  # (B, C, T) -> (B, E)

        # Early return if requested task requires no prediction heads
        if task == "encode":
            return out_features
        else:
            raise ValueError(f"Unknown task '{task}' requested for forward pass.")

    def _shared_step(self, batch: Union[Tensor, Tuple[Tensor, ...]], batch_idx: int) -> Dict[str, Tensor]:
        # Shared step for training, validation and testing
        metrics = {}
        losses = []
        if self.contrastive_losses is not None:
            metrics.update(self._contrastive_shared_step(batch, num_classes=10))
            losses.append(metrics["s_loss"])

        # Compute the sum of the (weighted) losses
        metrics["loss"] = sum(losses)

        return metrics

    def _contrastive_shared_step(
        self,
        batch: Union[Tensor, Tuple[Tensor, ...]],
        num_classes: int,
    ) -> Dict[str, Tensor]:
        # Forward pass through the encoder without gradient computation to fine-tune only the prediction heads
        time_series_input, target_labels = batch  # (N, C*T), (N,)

        # Squeeze dataset dimension
        time_series_input = time_series_input.squeeze(dim=0) # -> (B, C, T)
        seq_len = time_series_input.shape[-1]

        augmentation_1 = RandomCropResize(crop_rate_range=[0, 0.2], size=seq_len)
        augmentation_2 = RandomCropResize(crop_rate_range=[0, 0.2], size=seq_len)

        ts_augmented_1 = augmentation_1(time_series_input)  # (B, C, T)
        ts_augmented_2 = augmentation_2(time_series_input)  # (B, C, T)

        embedding_1 = self.encode(
            ts_input=ts_augmented_1,
            y_input=target_labels,
        )
        embedding_2 = self.encode(
            ts_input=ts_augmented_2,
            y_input=target_labels,
        )


        # Compute the loss/metrics for each target label, ignoring items for which targets are missing
        losses, metrics = {}, {}
        
        # if self.trainer.training:
        #     stage = "train_metrics"
        # elif self.trainer.validating:
        #     stage = "val_metrics"
        # else:
        #     stage = "test_metrics"

        for contrastive_task, contrastive_loss in self.contrastive_losses.items():

            # Compute loss for the entire batch at once
            loss_val = contrastive_loss(embedding_1, embedding_2)

            loss_name = f"{contrastive_loss.__class__.__name__.lower().replace('loss', '')}/{contrastive_task}"
            losses[loss_name] = loss_val

            # # Metrics are automatically updated inside the Metric objects
            # self.metrics[stage][contrastive_task].update(y_hat_flat, target_flat)

        losses["s_loss"] = torch.stack(list(losses.values())).mean()
        metrics.update(losses)

        return metrics

    # def on_test_epoch_end(self):
    #     output_data = []

    #     for target_task, collection in self.metrics["test_metrics"].items():
    #         # compute() returns a dict of results for this task
    #         results = collection.compute()

    #         for metric_name, value in results.items():
    #             tag = f"{metric_name}/{target_task}"
    #             self.log(f"test_{tag}", value)  # Log to logger
    #             output_data.append({"metric": tag, "value": value.item()})

    #         # Reset is handled by Lightning if logged, but manual reset is safe here
    #         collection.reset()

    #     # Save CSV once at the very end
    #     with open("test_metrics.csv", mode="w", newline="") as csv_file:
    #         writer = csv.DictWriter(csv_file, fieldnames=["metric", "value"])
    #         writer.writeheader()
    #         writer.writerows(output_data)
    #     logger.info("Test metrics saved to test_metrics.csv")
