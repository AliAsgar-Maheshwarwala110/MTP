import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback


class ResultArtifactCallback(Callback):
    """Persist epoch metrics and simple learning curves for reproducible runs."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.history = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        row = {"epoch": int(trainer.current_epoch)}
        for name, value in trainer.callback_metrics.items():
            if "/" in name:
                try:
                    row[name] = float(value.detach().cpu())
                except (AttributeError, TypeError, ValueError):
                    continue
        if row not in self.history:
            self.history.append(row)

    def on_fit_end(self, trainer, pl_module):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(self.history, f, indent=2)
        keys = sorted({key for row in self.history for key in row if key != "epoch"})
        with open(self.output_dir / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", *keys])
            writer.writeheader()
            writer.writerows(self.history)

        pairs = [
            ("train/loss", "val/loss", "loss"),
            ("train/f1", "val/f1", "f1"),
            ("train/auc", "val/auc", "auc"),
        ]
        for train_key, val_key, name in pairs:
            if not any(train_key in row or val_key in row for row in self.history):
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            for key, label in ((train_key, "train"), (val_key, "validation")):
                values = [row.get(key) for row in self.history]
                if any(value is not None for value in values):
                    ax.plot([row["epoch"] for row in self.history], values, label=label)
            ax.set(xlabel="Epoch", ylabel=name.upper(), title=f"E7 global normalization: {name}")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(self.output_dir / f"training_curve_{name}.png", dpi=160)
            plt.close(fig)
