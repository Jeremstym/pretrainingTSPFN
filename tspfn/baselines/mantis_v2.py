import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import csv
import pytorch_lightning as pl
from torchmetrics import MetricCollection, ConfusionMatrix
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassAveragePrecision,
    MulticlassCohenKappa,
    MulticlassRecall,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryAUROC,
    BinaryAccuracy,
    BinaryCohenKappa,
    BinaryRecall,
)

from mantis.architecture import MantisV2
from mantis.trainer import MantisTrainer
from sklearn.ensemble import RandomForestClassifier


def resize(X):
    X_scaled = F.interpolate(torch.tensor(X, dtype=torch.float), size=512, mode="linear", align_corners=False)
    return X_scaled


class Mantis2_SOTA(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 10,
        num_channels: int = 1,
        mantis_params: dict = None,
        rf_params: dict = None,
        finetuning: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.finetuning = finetuning
        print(f"num_classes: {num_classes}, num_channels: {num_channels}")
        # Optimized Mantis params for large tabular data
        self.mantis_params = mantis_params
        self.rf_params = rf_params
        self.num_patches = self.mantis_params.get("num_patches", 32)
        self.encoder_device = self.mantis_params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        network = MantisV2(**self.mantis_params)
        network = network.from_pretrained("paris-noah/MantisV2")
        self.encoder = MantisTrainer(device=self.encoder_device, network=network)
        self.clf = RandomForestClassifier(**self.rf_params)  # Using Random Forest as the classifier

        self.configure_metrics(device="cpu")  # Initialize metrics for both binary and multiclass

    def configure_metrics(self, device):
        # Binary classification metrics
        binary_metrics_template = MetricCollection(
            {
                "acc": BinaryAccuracy(),
                "auroc": BinaryAUROC(),
                "f1": BinaryF1Score(),
                "auprc": BinaryAveragePrecision(),
                "cohen_kappa": BinaryCohenKappa(),
                "recall": BinaryRecall(),
                "confusion_matrix": ConfusionMatrix(task="binary", num_classes=2),
            }
        )
        self.metrics_binary = torch.nn.ModuleDict({"test_metrics": binary_metrics_template.clone(prefix="test/")})
        # Multiclass classification metrics
        metrics_template = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=self.num_classes),
                "auroc": MulticlassAUROC(num_classes=self.num_classes),
                "f1": MulticlassF1Score(num_classes=self.num_classes, average="macro"),
                "auprc": MulticlassAveragePrecision(num_classes=self.num_classes),
                "cohen_kappa": MulticlassCohenKappa(num_classes=self.num_classes),
                "recall": MulticlassRecall(num_classes=self.num_classes, average="macro"),
                "confusion_matrix": ConfusionMatrix(task="multiclass", num_classes=self.num_classes),
            }
        )
        self.metrics = torch.nn.ModuleDict({"test_metrics": metrics_template.clone(prefix="test/")})

        # Set to device
        self.metrics = self.metrics.to(device)
        self.metrics_binary = self.metrics_binary.to(device)

    def setup(self, stage=None):
        # Trigger fitting for both fit and validate stages
        assert stage in ["fit", "validate", "test"], f"Unexpected stage: {stage}"
        print("--- Fitting Mantis on Training Data ---")
        # Access the underlying train_dataloader from the datamodule
        train_loader = self.trainer.datamodule.train_dataloader()

        all_x, all_y = [], []
        for batch in train_loader:
            # Handle if train_loader is also a CombinedLoader or simple tuple
            x, y = batch if isinstance(batch, (tuple, list)) else batch["train"]
            batch_size, num_channels, seq_len = x.shape
            if seq_len % self.num_patches != 0:
                x = resize(x)  # Resize to ensure divisibility by num_patches
            all_x.append(x)
            all_y.append(y)

        X_train = torch.cat(all_x, dim=0)
        y_train = torch.cat(all_y, dim=0)
        print(f"--- Training Data Loaded: {X_train.shape[0]} samples ---")
        print(f"Mantis Parameters: {self.mantis_params}")

        if not self.finetuning:
            print(f"Support shape: {X_train.shape}, Labels shape: {y_train.shape}")
            X_train = self.encoder.transform(X_train.to(self.encoder_device))
            print(f"--- Mantis Embedding Complete: {X_train.shape[0]} samples ---")
            y_train = y_train.cpu().numpy() # Convert to numpy for Random Forest
            self.clf.fit(X_train, y_train)
            print(f"--- Random Forest Fit Complete ({len(X_train)} samples) ---")
        else:
            print("Fine-tuning Mantis on training data...")
            self.encoder.fit(X_train.to(self.encoder_device), y_train)
            print("Fine-tuning complete.")

            print("Extracting fine-tuned embeddings for Random Forest...")
            X_train_embedded = self.encoder.transform(X_train.to(self.encoder_device))
            self.clf.fit(X_train_embedded, y_train)
            print("--- Random Forest Fit Complete ---")

        num_classes = np.unique(y_train).shape[0]
        if self.num_classes != num_classes:
            self.num_classes = num_classes
            self.configure_metrics(device="cpu")  # Reconfigure metrics if number of classes has changed

    def test_step(self, batch, batch_idx):
        if not self.finetuning:
            batch_dict, _, _ = batch if isinstance(batch, (tuple, list)) else (batch, None, None)
            if "val" not in batch_dict:
                raise ValueError("Expected 'val' key in batch for validation data.")
            x, y = batch_dict["val"]
        else:
            x, y = batch

        print(f"Original test batch shape: {x.shape}, labels shape: {y.shape}")
        batch_size, num_channels, seq_len = x.shape
        if seq_len % self.num_patches != 0:
            x = resize(x)  # Resize to ensure divisibility by num_patches

        if not self.finetuning:
            x_eval = self.encoder.transform(x.to(self.encoder_device))

            print(f"Query shape for Random Forest: {x_eval.shape}, Labels shape: {y.shape}")
            y_probs = self.clf.predict_proba(x_eval)
            y_probs_ts = torch.tensor(y_probs, device=self.device)
        else:
            x_eval = self.encoder.transform(x.to(self.encoder_device))
            print(f"Query shape for Random Forest (fine-tuned): {x_eval.shape}, Labels shape: {y.shape}")
            y_probs = self.encoder.predict_proba(x_eval)
            y_probs_ts = torch.tensor(y_probs, device=self.device)

        print(f"--- Test Step {batch_idx}: Evaluated {x_eval.shape[0]} samples ---")

        if self.num_classes == 2:
            # For binary classification, use probabilities of the positive class
            y_probs_ts = torch.softmax(y_probs_ts, dim=-1)[:, 1]
            self.metrics_binary["test_metrics"].update(y_probs_ts, y.long())
        elif self.num_classes > 2:
            # For multiclass classification, use the full probability distribution
            self.metrics["test_metrics"].update(y_probs_ts, y)

    def on_test_epoch_end(self):
        output_data = []
        if self.num_classes == 2:
            metrics_collection = self.metrics_binary
        else:
            metrics_collection = self.metrics
        results = metrics_collection["test_metrics"].compute()
        for metric_name, value in results.items():
            if "confusion_matrix" not in metric_name:  # Skip confusion matrix for logging
                self.log(metric_name, value, prog_bar=True, on_epoch=True, on_step=False)
                value = value.item() if isinstance(value, torch.Tensor) else value
            output_data.append({"metric": metric_name, "value": value})

        metrics_collection["test_metrics"].reset()  # Reset metrics after logging

        with open("mantis_rf_test_metrics.csv", mode="w", newline="") as csvfile:
            fieldnames = ["metric", "value"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in output_data:
                writer.writerow(row)
        # # Initialize a custom attribute to store predictions
        # self.test_predictions_storage = []

        # for batch in self.trainer.datamodule.test_dataloader():
        #     batch_dict, _, _ = batch if isinstance(batch, (tuple, list)) else (batch, None, None)

        #     x_test, y_test = batch_dict["val"]
        #     if x_test.shape[2] % self.num_patches != 0:
        #         x_test = resize(x_test)  # Resize to ensure divisibility by num_patches
        #     x_eval_test = self.encoder.transform(x_test.to(self.encoder_device))
        #     y_probs_test = self.clf.predict_proba(x_eval_test)
        #     y_probs_ts_test = torch.tensor(y_probs_test, device=self.device)

        #     # Store predictions and true labels for later use
        #     self.test_predictions_storage.append({"probs": y_probs_ts_test.cpu(), "targets": y_test.cpu()})

        # df = pd.DataFrame(self.test_predictions_storage)
        # df.to_csv("mantis_rf_test_predictions.csv", index=False)

        # self.test_predictions_storage.clear()

    def configure_optimizers(self):
        # Dummy optimizer since Random Forest isn't trained via AdamW
        return torch.optim.Adam(self.parameters(), lr=1e-3)
