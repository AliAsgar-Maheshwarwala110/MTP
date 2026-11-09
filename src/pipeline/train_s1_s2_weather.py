"""Train one controlled lightweight S1/S2/NASA POWER ablation."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import seed_everything
from torchmetrics.classification import BinaryAUROC
from torch.utils.data import DataLoader

from src.data.s1_s2_weather_dataset import load_aligned_data
from src.models.lightweight_s1_s2_weather import (
    LightweightS1S2WeatherClassifier,
    parameter_count,
)


LOG = logging.getLogger("e8-training")


def _append_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as report:
        fcntl.flock(report.fileno(), fcntl.LOCK_EX)
        report.write(content)
        report.flush()
        fcntl.flock(report.fileno(), fcntl.LOCK_UN)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predicted = probabilities >= 0.5
    labels = labels.bool()
    tp = (predicted & labels).sum().item()
    fp = (predicted & ~labels).sum().item()
    fn = (~predicted & labels).sum().item()
    return {
        "accuracy": float((predicted == labels).float().mean().item()),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "auc": float(BinaryAUROC()(probabilities.cpu(), labels.int().cpu()).item()),
    }


def _run_epoch(model, loader, device, optimizer=None, epoch=0, total_epochs=0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    logits_all = []
    labels_all = []
    started = time.monotonic()
    for batch_index, batch in enumerate(loader, start=1):
        batch = _move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, batch["label"])
            if training:
                loss.backward()
                optimizer.step()
        batch_size = batch["label"].shape[0]
        total_loss += loss.detach().item() * batch_size
        logits_all.append(logits.detach())
        labels_all.append(batch["label"].detach())
        if training and batch_index % 50 == 0:
            elapsed = time.monotonic() - started
            remaining = elapsed / batch_index * (len(loader) - batch_index)
            LOG.info(
                "epoch %d/%d batch %d/%d elapsed %.1f min ETA %.1f min",
                epoch, total_epochs, batch_index, len(loader),
                elapsed / 60, remaining / 60,
            )
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    metrics = _metrics(logits, labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def _atomic_torch_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    weather_dir = args.weather_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, workers=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this run requires a GPU")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda")
    LOG.info("Using %s (%s)", torch.cuda.get_device_name(device), device)

    datasets, image_stats, weather_stats, splits = load_aligned_data(
        root, weather_dir, args.stats_samples, args.ablation
    )
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        for split, dataset in datasets.items()
    }
    weather_features = 27
    weather_sequence_features = 11
    model = LightweightS1S2WeatherClassifier(
        ablation=args.ablation,
        weather_features=weather_features,
        weather_sequence_features=weather_sequence_features,
        pretrained=not args.no_pretrained,
    ).to(device)
    LOG.info(
        "Model parameters: %s; aligned samples: %s; weather features: %d",
        f"{parameter_count(model):,}",
        {split: len(dataset) for split, dataset in datasets.items()},
        weather_features,
    )
    positive = sum(entry["pixel_annotated"] > 0 for entry in splits["train"])
    LOG.info(
        "Training labels: %d positive / %d negative",
        positive, len(splits["train"]) - positive,
    )

    config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "pretrained": not args.no_pretrained,
        "ablation": args.ablation,
        "best_checkpoint_metric": "validation F1 at threshold 0.5",
        "task": "patch-level landslide detection",
        "weather_timing": "strictly before inventory event date",
        "weather_fusion": (
            "30-day bidirectional GRU with learned temporal attention, "
            "event-relative aggregate MLP, transformer cross-modal fusion, and sigmoid gate"
            if args.ablation == "s1_s2_weather" else "disabled"
        ),
        "aligned_split_counts": {split: len(rows) for split, rows in splits.items()},
        "parameter_count": parameter_count(model),
        "image_statistics": image_stats,
        "weather_statistics": weather_stats,
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2))
    if args.report_file:
        _append_report(
            args.report_file,
            f"\n### {args.ablation} arm progress\n\n"
            f"- Started: {datetime.now(timezone.utc).isoformat()}; "
            f"parameters: {parameter_count(model):,}; output: `{output}`.\n",
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_f1 = -1.0
    best_epoch = 0
    patience_used = 0
    history = []
    training_started = time.monotonic()
    best_path = output / "best_checkpoint.pt"

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        train_metrics = _run_epoch(
            model, loaders["train"], device, optimizer, epoch, args.epochs
        )
        val_metrics = _run_epoch(model, loaders["val"], device)
        epoch_minutes = (time.monotonic() - epoch_started) / 60
        history.append({
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "epoch_minutes": epoch_minutes,
        })
        elapsed = (time.monotonic() - training_started) / 60
        avg_epoch = elapsed / epoch
        eta = avg_epoch * max(args.epochs - epoch, 0)
        LOG.info(
            "epoch %d/%d train loss %.4f F1 %.4f AUC %.4f; "
            "val loss %.4f F1 %.4f AUC %.4f; epoch %.1f min; "
            "training elapsed %.1f min ETA %.1f min",
            epoch, args.epochs,
            train_metrics["loss"], train_metrics["f1"], train_metrics["auc"],
            val_metrics["loss"], val_metrics["f1"], val_metrics["auc"],
            epoch_minutes, elapsed, eta,
        )
        if args.report_file:
            _append_report(
                args.report_file,
                f"- {args.ablation} epoch {epoch}/{args.epochs}: "
                f"train F1 {train_metrics['f1']:.4f}, AUC {train_metrics['auc']:.4f}; "
                f"validation F1 {val_metrics['f1']:.4f}, AUC {val_metrics['auc']:.4f}; "
                f"elapsed {elapsed:.1f} min, ETA {eta:.1f} min.\n",
            )
        with (output / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            patience_used = 0
            _atomic_torch_save(
                {"model_state_dict": model.state_dict(), "epoch": epoch,
                 "validation_metrics": val_metrics, "config": config},
                best_path,
            )
            LOG.info("Saved new best checkpoint: %s", best_path)
        else:
            patience_used += 1
            if patience_used >= args.patience:
                LOG.info("Early stopping after %d epochs without validation F1 improvement", args.patience)
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = _run_epoch(model, loaders["test"], device)
    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_epoch": best_epoch,
        "best_validation_f1": best_f1,
        "test_metrics": test_metrics,
        "training_minutes": (time.monotonic() - training_started) / 60,
        "parameter_count": parameter_count(model),
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "weather_features": weather_features,
        "weather_sequence_shape": [30, weather_sequence_features] if args.ablation == "s1_s2_weather" else None,
        "event_count": len({entry["inventory"] for rows in splits.values() for entry in rows}),
        "event_generalization_note": (
            "Official aligned random patch splits include the same inventories across splits; "
            "test scores are not an unseen-event estimate."
        ),
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    LOG.info("Final test metrics using best validation checkpoint: %s", test_metrics)
    LOG.info("Run artifacts: %s", output)
    if args.report_file:
        _append_report(
            args.report_file,
            f"\n- {args.ablation} completed {result['completed_at_utc']}: "
            f"best epoch {best_epoch}, validation F1 {best_f1:.4f}; test accuracy "
            f"{test_metrics['accuracy']:.4f}, AUC {test_metrics['auc']:.4f}, "
            f"F1 {test_metrics['f1']:.4f}; training {result['training_minutes']:.1f} min; "
            f"parameters {result['parameter_count']:,}; artifacts `{output}`.\n",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--weather-dir", type=Path, default=Path("environmental_data/e8_s1_s2_power"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-file", type=Path)
    parser.add_argument(
        "--ablation",
        choices=("s2", "s1_s2", "s1_s2_weather"),
        default="s1_s2_weather",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--stats-samples", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train(args)


if __name__ == "__main__":
    main()
