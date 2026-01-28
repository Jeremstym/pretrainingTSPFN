import torch
import numpy as np
import xgboost as xgb
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy, 
    MulticlassAUROC, 
    MulticlassF1Score, 
    MulticlassAveragePrecision
)

class XGBoostStaticBaseline(pl.LightningModule):
    def __init__(self, num_classes: int = 10, xgb_params: dict = None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        
        # ... (xgb_params init remains the same) ...

        # Define metrics once, we will clone them for Val and Test
        metrics = MetricCollection({
            "acc": MulticlassAccuracy(num_classes=num_classes),
            "auroc": MulticlassAUROC(num_classes=num_classes),
            "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
            "auprc": MulticlassAveragePrecision(num_classes=num_classes)
        })
        
        self.val_metrics = metrics.clone(prefix="val_xgb_")
        self.test_metrics = metrics.clone(prefix="test_xgb_") # Added for test stage
        self.clf = None

    # ... (setup, validation_step remain the same) ...

    def test_step(self, batch, batch_idx):
        if self.clf is None: return
        x, y = batch
        x_eval = x.view(-1, x.size(-1)).cpu().numpy()
        y_eval = y.view(-1).cpu()
        
        y_probs = self.clf.predict_proba(x_eval)
        y_probs_ts = torch.tensor(y_probs, device=self.device)
        
        # USE TEST METRICS HERE
        self.test_metrics.update(y_probs_ts, y_eval.to(self.device))

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