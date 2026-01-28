import torch
import numpy as np
import xgboost as xgb
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassAveragePrecision,
)


class XGBoostStaticBaseline(pl.LightningModule):
    def __init__(self, num_classes: int = 10, xgb_params: dict = None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes

        # ... (xgb_params init remains the same) ...

        # Define metrics once, we will clone them for Val and Test
        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes),
                "auroc": MulticlassAUROC(num_classes=num_classes),
                "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
                "auprc": MulticlassAveragePrecision(num_classes=num_classes),
            }
        )

        self.val_metrics = metrics.clone(prefix="val_xgb_")
        self.test_metrics = metrics.clone(prefix="test_xgb_")  # Added for test stage
        self.clf = None

    # ... (setup, validation_step remain the same) ...

    def on_test_start(self):
        """Ensures XGBoost is fitted specifically before testing starts."""
        if self.clf is None:
            print("--- Fitting XGBoost on-the-fly for Testing ---")
            self._fit_xgboost()

    def _fit_xgboost(self):
        # Move the fitting logic to a private method to keep it clean
        train_loader = self.trainer.datamodule.train_dataloader()

        all_x, all_y = [], []
        for batch in train_loader:
            x, y = batch
            all_x.append(x.view(-1, x.size(-1)).cpu())
            all_y.append(y.view(-1).cpu())

        X_train = torch.cat(all_x, dim=0).numpy()
        y_train = torch.cat(all_y, dim=0).numpy()

        self.clf = xgb.XGBClassifier(**self.xgb_params)
        self.clf.fit(X_train, y_train)

    def test_step(self, batch, batch_idx):
        if self.clf is None:
            return
        x, y = batch
        x_eval = x.view(-1, x.size(-1)).cpu().numpy()
        y_eval = y.view(-1).cpu()

        y_probs = self.clf.predict_proba(x_eval)
        y_probs_ts = torch.tensor(y_probs, device=self.device)

        # Log every step (Lightning will handle the averaging)
        self.test_metrics.update(y_probs_ts, y_eval.to(self.device))

        # NEW: Log individual steps to ensure the logger is "awake"
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        # compute returns a dict of results
        output = self.test_metrics.compute()
        # Log and set sync_dist=True if using multiple GPUs
        self.log_dict(output, prog_bar=True, logger=True)
        self.test_metrics.reset()

        # OPTIONAL: Print results to console immediately so you don't have to wait for logs
        print("\n--- XGBoost Test Results ---")
        for k, v in output.items():
            print(f"{k}: {v:.4f}")
