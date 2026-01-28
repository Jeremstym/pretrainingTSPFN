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
    def __init__(self, num_classes: int = 10, xgb_params: dict = None):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        
        # Optimized XGBoost params for large tabular data
        self.xgb_params = xgb_params or {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "tree_method": "hist",     # Change to "gpu_hist" for GPU speedup
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "verbosity": 0
        }
        
        # Metric Collection for clean evaluation
        metrics = MetricCollection({
            "acc": MulticlassAccuracy(num_classes=num_classes),
            "auroc": MulticlassAUROC(num_classes=num_classes),
            "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
            "auprc": MulticlassAveragePrecision(num_classes=num_classes)
        })
        
        self.val_metrics = metrics.clone(prefix="val_xgb_")
        self.clf = None

    def setup(self, stage=None):
        # Fit if we are in 'fit' (training) OR 'test' stage
        if stage in ["fit", "test"] or stage is None:
            if self.clf is None:  # Prevent re-fitting if already done
                print(f"--- Fitting XGBoost for stage: {stage} ---")
                # We still fit on the TRAINING data even if we are testing
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

    def validation_step(self, batch, batch_idx):
        if self.clf is None:
            return
            
        x, y = batch # x: [B, N, P], y: [B, N]
        
        # Flatten batch and sequence for evaluation
        x_eval = x.view(-1, x.size(-1)).cpu().numpy()
        y_eval = y.view(-1).cpu() # Keep as tensor for metrics
        
        # Predict Probabilities: Shape (Batch*N, Num_Classes)
        y_probs = self.clf.predict_proba(x_eval)
        y_probs_ts = torch.tensor(y_probs, device=self.device)
        
        # Update metric collection
        self.val_metrics.update(y_probs_ts, y_eval.to(self.device))

    def on_validation_epoch_end(self):
        # Compute and log all metrics at once
        output = self.val_metrics.compute()
        self.log_dict(output, prog_bar=True)
        self.val_metrics.reset()

    def configure_optimizers(self):
        # Dummy optimizer since XGBoost isn't trained via SGD
        return torch.optim.Adam(self.parameters(), lr=1e-3)