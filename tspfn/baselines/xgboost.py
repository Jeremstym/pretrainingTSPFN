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
        self.clf = None  # This starts as None
        
        # ... (metric init same as before) ...

    def _ensure_fitted(self):
        """Forces XGBoost to fit if it hasn't already."""
        if self.clf is not None:
            return

        print("--- [FORCE] Fitting XGBoost Baseline ---")
        
        # Access the datamodule via the trainer
        dm = getattr(self.trainer, "datamodule", None)
        if dm is None:
            raise RuntimeError("No DataModule found on trainer. Pass it: trainer.test(model, datamodule=dm)")

        # Manually trigger setup on the datamodule just in case
        dm.setup(stage="fit")
        loader = dm.train_dataloader()
        
        all_x, all_y = [], []
        print(f"--- Collecting training batches... ---")
        for batch in loader:
            x, y = batch # x: [B, N, P], y: [B, N]
            all_x.append(x.view(-1, x.size(-1)).cpu())
            all_y.append(y.view(-1).cpu())
        
        if not all_x:
            raise RuntimeError("Training loader returned no data! Check your dataset paths.")

        X_train = torch.cat(all_x, dim=0).numpy()
        y_train = torch.cat(all_y, dim=0).numpy()
        
        print(f"--- Fitting XGBoost on {X_train.shape[0]} samples ---")
        self.clf = xgb.XGBClassifier(**self.hparams.xgb_params)
        self.clf.fit(X_train, y_train)
        print("--- Fitting Complete! ---")

    def on_test_start(self):
        # This hook runs AFTER setup but BEFORE test_step
        self._ensure_fitted()

    def on_validation_start(self):
        # This hook runs BEFORE validation_step
        self._ensure_fitted()

    def test_step(self, batch, batch_idx):
        # Since _ensure_fitted was called in on_test_start, self.clf is now ready
        x, y = batch
        x_eval = x.view(-1, x.size(-1)).cpu().numpy()
        
        y_probs = self.clf.predict_proba(x_eval)
        y_probs_ts = torch.tensor(y_probs, device=self.device)
        
        self.test_metrics.update(y_probs_ts, y.view(-1).to(self.device))
        
    def on_test_epoch_end(self):
        results = self.test_metrics.compute()
        print("\n" + "="*30 + "\nTEST RESULTS:\n", results, "\n" + "="*30)
        self.log_dict(results)
        self.test_metrics.reset()