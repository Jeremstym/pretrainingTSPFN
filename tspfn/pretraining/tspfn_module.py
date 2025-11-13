import logging
from sklearn.model_selection import train_test_split
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, cast

import hydra
from omegaconf import DictConfig
import torch
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor, nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryAveragePrecision,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassF1Score,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics import MetricCollection

from data.utils.decorators import auto_move_data
from tspfn.system import TSPFNSystem

logger = logging.getLogger(__name__)


class TSPFNPretraining(TSPFNSystem):
    """Multi-modal transformer to learn a representation from cardiac imaging and patient records data."""

    def __init__(
        self,
        embed_dim: int,
        split_finetuning: float = 0.5,
        predict_losses: Optional[Dict[str, Callable[[Tensor, Tensor], Tensor]] | DictConfig] = None,
        *args,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            embed_dim: Size of the tokens/embedding for all the modalities.
            predict_losses: Supervised criteria to measure the error between the predicted attributes and their real
                value.
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        # Ensure string tags are converted to their appropriate enum types
        # And do it before call to the parent's `init` so that the converted values are saved in `hparams`

        super().__init__(*args, **kwargs)

        # Add shortcut to lr to work with Lightning's learning rate finder
        self.hparams["lr"] = None

        # Configure losses/metrics to compute at each train/val/test step
        self.metrics = nn.ModuleDict()
        # if target in TabularAttribute.numerical_attrs():
        #     self.metrics[target] = MetricCollection([MeanAbsoluteError(), MeanSquaredError()])
        # elif target in TabularAttribute.binary_attrs():
        #     self.metrics[target] = MetricCollection(
        #         [
        #             BinaryAccuracy(),
        #             BinaryAUROC(),
        #             BinaryAveragePrecision(),
        #             BinaryF1Score(),
        #         ],
        #     )
        # else:  # attr in TabularAttribute.categorical_attrs()

        # Supervised losses and metrics
        self.predict_losses = {}
        if predict_losses:
            self.predict_losses = {
                target_task: (
                    hydra.utils.instantiate(target_loss) if isinstance(target_loss, DictConfig) else target_loss
                )
                for target_task, target_loss in predict_losses.items()
            }

        # Initialize transformer encoder and self-supervised + prediction heads
        self.encoder, self.prediction_heads = self.configure_model()

        # Initialize inference storage tensors
        self.ts_train_for_inference = torch.Tensor().to(self.device)
        self.y_train_for_inference = torch.Tensor().to(self.device)

    @property
    def example_input_array(self) -> Tensor:
        """Redefine example input array based on the cardiac attributes provided to the model."""
        # 2 is the size of the batch in the example
        labels = torch.randperm(10)
        time_series_attrs = torch.randn(10, 64)
        ts_example_input = torch.cat([time_series_attrs, labels.unsqueeze(1)], dim=1)
        return ts_example_input

    def configure_model(
        self,
    ) -> Tuple[nn.Module, nn.Module, Optional[nn.ModuleDict]]:
        """Build the model, which must return a transformer encoder, and self-supervised or prediction heads."""
        # Build the transformer encoder
        encoder = hydra.utils.instantiate(self.hparams["model"]["encoder"])

        # Build the prediction heads following the architecture proposed in
        # https://arxiv.org/pdf/2106.11959
        prediction_heads = None
        if self.predict_losses:
            prediction_heads = nn.ModuleDict()
            for target_task in self.predict_losses:
                prediction_heads[target_task] = hydra.utils.instantiate(
                    self.hparams["model"]["prediction_head"],
                )

        return encoder, prediction_heads

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure optimizer to ignore parameters that should remain frozen (e.g. tokenizers)."""
        # Frozen tabpfn encoder
        if self.hparams["model"].get("freeze_encoder", False):
            for param in self.encoder.parameters():
                param.requires_grad = False
        return super().configure_optimizers()

    @auto_move_data
    def process_data(
        self,
        time_series_attrs: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Tokenizes the input time-series attributes, providing a mask of non-missing attributes.

        Args:
            time_series_attrs: (N, T), Batch of T-lengthed time series.

        Returns:
            - (S, 1), Support set labels.
            - (Q, 1), Query set labels.
            - (N (=S+Q), T), Time series input for .
        """

        # Tokenize the attributes
        assert time_series_attrs is not None, "At least time_series_attrs must be provided to process_data."

        ts = time_series_attrs[:, :-1]
        indices = torch.arange(ts.shape[0])
        y = time_series_attrs[:, -1]

        if self.training or len(self.y_train_for_inference) == 0:
            assert self.hparams["split_finetuning"] > 0.0, "split_finetuning must be > 0.0 when training."
            label = y.clone().cpu()
            try:
                train_indices, test_indices = train_test_split(
                    indices,
                    test_size=self.hparams["split_finetuning"],
                    random_state=self.hparams["seed"],
                    stratify=label,
                )
            except ValueError:
                if len(indices) == 1:
                    print("Sanity check with single sample for TS, skipping train/test split.")
                    train_indices, test_indices = indices, indices
                else:
                    print("Stratified splitting failed, performing non-stratified split instead.")
                    train_indices, test_indices = train_test_split(
                        indices,
                        test_size=self.hparams["split_finetuning"],
                        random_state=self.hparams["seed"],
                    )

            ts_support = torch.as_tensor(ts[train_indices], dtype=torch.float32)
            ts_query = torch.as_tensor(ts[test_indices], dtype=torch.float32)

            ts = torch.cat([ts_support, ts_query], dim=0)

        if self.training or len(self.y_train_for_inference) == 0:
            y_batch_support = torch.as_tensor(y[train_indices], dtype=torch.float32)
            y_batch_query = torch.as_tensor(y[test_indices], dtype=torch.float32)
        else:
            y_batch_support = y.to(self.device)
            y_batch_query = y.to(self.device)

        if ts.ndim == 2:
            ts = ts.unsqueeze(1)

        return (
            y_batch_support,
            y_batch_query,
            ts,
        )

    @auto_move_data
    def encode(
        self,
        y_batch_support: Tensor,
        ts: Tensor,
    ) -> Tensor:
        """Embeds input sequences using the encoder model, optionally selecting/pooling output ts for the embedding.

        Args:
            y_batch_support: (S, 1), Support set labels.
            ts: (N (=S+Q), T), Tokens to feed to the encoder.
        Returns: (Q, E), Embeddings of the input sequences.
        """

        if self.training or len(self.ts_train_for_inference) == 0:
            out_features = self.encoder(ts, y_batch_support)[:, -1, :]

        else:
            # Use train set as context for predicting the query set on val/test inference
            ts = torch.cat([self.ts_train_for_inference, ts], dim=0)
            y_train = self.y_train_for_inference
            out_features = self.encoder(ts, y_train)[:, -1, :]

        return out_features  # (N, d_model)

    @auto_move_data
    def forward(
        self,
        time_series_attrs: Tensor,
        task: Literal["encode", "predict"] = "encode",
    ) -> Tensor | Dict[str, Tensor]:
        """Performs a forward pass through i) the tokenizer, ii) the transformer encoder and iii) the prediction head.

        Args:
            time_series_attrs: (N, T): time series inputs.
            task: Flag indicating which type of inference task to perform.

        Returns:
            if `task` == 'encode':
                (Q, E), Batch of features extracted by the encoder.
            if `task` == 'predict' (and the model includes prediction heads):
                1 * (Q), Prediction for each target in `losses`.
        """
        if task != "encode" and not self.prediction_heads:
            raise ValueError(
                "You requested to perform a prediction task, but the model does not include any prediction heads."
            )
        y_batch_support, y_batch_query, ts = self.process_data(
            time_series_attrs,
        )  # (N, S, E), (N, S)

        out_features = self.encode(y_batch_support, ts)  # (N, S, E) -> (N, E)

        # Early return if requested task requires no prediction heads
        if task == "encode":
            return out_features

        elif task == "predict":
            assert (
                self.prediction_heads is not None
            ), "You requested to perform a prediction task, but the model does not include any prediction heads."

            # Forward pass through each target's prediction head
            n_class = torch.unique(torch.cat([y_batch_support, y_batch_query])).shape[0]
            predictions = {
                target_task: prediction_head(out_features, n_class) for target_task, prediction_head in self.prediction_heads.items()
            }

            # Squeeze out the singleton dimension from the predictions' features (only relevant for scalar predictions)
            predictions = {target_task: prediction.squeeze(dim=1) for target_task, prediction in predictions.items()}
            return predictions

        else:
            raise ValueError(f"Unknown task '{task}' requested for forward pass.")

    @auto_move_data
    def get_latent_vectors(
        self,
        batch: Tensor,
        batch_idx: int,
    ) -> Tensor:
        """Extracts the latent vectors from the encoder for the given batch."""
        y_batch_support, y_batch_query, ts = self.process_data(time_series_attrs=batch)  # (N, S, E), (N, S)
        return self.encode(
            y_batch_support,
            ts,
        )  # (N, S, E) -> (N, E)

    def _shared_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        # Extract time-series inputs from the batch

        y_batch_support, y_batch_query, ts = self.process_data(time_series_attrs=batch)  # (N, S, E), (N, S)

        metrics = {}
        losses = []
        if self.predict_losses is not None:
            metrics.update(
                self._prediction_shared_step(y_batch_support, y_batch_query, ts)
            )
            losses.append(metrics["s_loss"])

        # Compute the sum of the (weighted) losses
        metrics["loss"] = sum(losses)

        return metrics

    def _prediction_shared_step(
        self,
        y_batch_support: Tensor,
        y_batch_query: Tensor,
        ts: Tensor,
    ) -> Dict[str, Tensor]:
        # Forward pass through the encoder without gradient computation to fine-tune only the prediction heads
        assert (
            self.prediction_heads is not None
        ), "You requested to perform a prediction task, but the model does not include any prediction heads."
        prediction = self.encode(y_batch_support, ts)
        n_class = torch.unique(torch.cat([y_batch_support, y_batch_query])).shape[0]
        predictions = {}
        for target_task, prediction_head in self.prediction_heads.items():
            pred = prediction_head(prediction, n_class)
            predictions[target_task] = pred

        for target_task in self.predict_losses:
            self.metrics[target_task] = MetricCollection(
                [
                    MulticlassAccuracy(num_classes=n_class, average="micro"),
                    MulticlassAUROC(num_classes=n_class, average="macro"),
                    MulticlassAveragePrecision(num_classes=n_class, average="macro"),
                    MulticlassF1Score(num_classes=n_class, average="macro"),
                ]
            ).to(self.device)

        # Compute the loss/metrics for each target label, ignoring items for which targets are missing
        losses, metrics = {}, {}

        target_batch = y_batch_query

        for target_task, target_loss in self.predict_losses.items():
            target, y_hat = target_batch, predictions[target_task]

            target = target.float() if len(torch.unique(target)) == 2 else target.long()

            losses[f"{target_loss.__class__.__name__.lower().replace('loss', '')}/{target_task}"] = target_loss(
                y_hat,
                target,
            )

            for metric_tag, metric in self.metrics[target_task].items():
                metric.update(y_hat, target)

        losses["s_loss"] = torch.stack(list(losses.values())).mean()
        metrics.update(losses)

        return metrics

    def _on_epoch_start(self):
        train_loader = self.trainer.train_dataloader
        if train_loader is None:
            return "No training dataloader found while setting up inference storage tensors"
        else:
            if callable(train_loader):
                train_loader = train_loader()

            ts_batch_list = [ts_batch for ts_batch in train_loader]
            ts_tokens_support = torch.vstack(ts_batch_list)

            assert ts_tokens_support.ndim == 2, f"{ts_tokens_support.ndim=}, {ts_tokens_support.shape=}"

            self.ts_train_for_inference = ts_tokens_support.to(self.device)

    def on_validation_epoch_start(self):
        self._on_epoch_start()

    def on_test_epoch_start(self):
        out_message = self._on_epoch_start()
        if out_message is not None:
            logger.info(f"Test epoch start: {out_message}")
            raise ValueError(out_message)

    def on_test_epoch_end(self):
        all_metrics = {}
        for target_task in self.predict_losses:
            for metric_tag, metric in self.metrics[target_task].items():
                metrics_value = metric.compute()
                self.log(f"test_{metric_tag}/{target_task}", metrics_value)
                all_metrics[f"{metric_tag}/{target_task}"] = (
                    metrics_value.item() if hasattr(metrics_value, "item") else metrics_value
                )
                metric.reset()

        # Print metrics to terminal
        logger.info(f"Test metrics: {all_metrics}")
