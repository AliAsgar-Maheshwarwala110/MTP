import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class LightweightLitModule(LightningModule):
    def __init__(self, net, learning_rate=1e-3, weight_decay=1e-4, pos_weight=1.0):
        super().__init__()
        self.net = net
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.train_f1 = BinaryF1Score()
        self.train_auc = BinaryAUROC()
        self.train_accuracy = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_auc = BinaryAUROC()
        self.val_accuracy = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()
        self.test_auc = BinaryAUROC()
        self.test_accuracy = BinaryAccuracy()

    def forward(self, batch):
        return self.net(batch["s2"], batch["s1"])

    def _step(self, batch, stage):
        logits = self(batch)
        labels = batch["label"]
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.pos_weight.to(logits.device)
        )
        probs = torch.sigmoid(logits)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        metrics = {
            "f1": getattr(self, f"{stage}_f1"),
            "auc": getattr(self, f"{stage}_auc"),
            "accuracy": getattr(self, f"{stage}_accuracy"),
        }
        for name, metric in metrics.items():
            metric.update(probs, labels.int())
            self.log(f"{stage}/{name}", metric, on_step=False, on_epoch=True, prog_bar=True)
        if stage == "val":
            self.val_precision.update(probs, labels.int())
            self.val_recall.update(probs, labels.int())
            self.log("val/precision", self.val_precision, on_epoch=True)
            self.log("val/recall", self.val_recall, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
