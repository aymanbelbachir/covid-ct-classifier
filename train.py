"""Train MobileNetV3-Large on the CT slices and save the best model.

Usage:
    python train.py                       # group split (default), 25 epochs
    python train.py --split-mode slice    # original notebook split
    python train.py --epochs 5            # shorter run
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")              # save figures to files, no window needed
import matplotlib.pyplot as plt
import torch
from torch import nn, optim

from covid_ct import config
from covid_ct.data import get_dataloaders
from covid_ct.engine import evaluate, train_one_epoch
from covid_ct.model import build_model


def save_curves(history, path, title):
    """Plot the loss and accuracy of every epoch and save the figure."""
    epochs = range(1, len(history["val_acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, name in ((ax1, "loss", "Loss"), (ax2, "acc", "Accuracy")):
        ax.plot(epochs, history[f"train_{key}"], label="Train")
        ax.plot(epochs, history[f"val_{key}"], label="Validation")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the COVID-19 CT classifier.")
    parser.add_argument("--split-mode", choices=config.SPLIT_MODES, default=config.DEFAULT_SPLIT_MODE,
                        help="'group' keeps near-identical slices together, 'slice' is the original split")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="number of epochs")
    args = parser.parse_args()

    model_path = config.model_path(args.split_mode)
    history_path = config.history_path(args.split_mode)
    curves_path = config.curves_path(args.split_mode)

    if model_path.exists():
        answer = input(f"{config.short(model_path)} already exists and will be overwritten. Continue? [y/N] ")
        if answer.strip().lower() != "y":
            print("Cancelled.")
            return

    torch.manual_seed(config.SEED)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.ASSETS_DIR.mkdir(exist_ok=True)

    device = config.get_device()
    loaders = get_dataloaders(["train", "val"], mode=args.split_mode)
    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    print(f"Training on {device} for {args.epochs} epochs with the {args.split_mode} split | "
          f"{len(loaders['train'].dataset):,} train / {len(loaders['val'].dataset):,} validation images")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)

        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)

        saved = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            saved = "  <- best, saved"

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.2%} | "
              f"val loss {val_loss:.4f} acc {val_acc:.2%} | "
              f"{time.time() - t0:.0f}s{saved}")

        # saved after every epoch, so nothing is lost if the run is interrupted
        with open(history_path, "w") as f:
            json.dump(history, f)

    save_curves(history, curves_path, f"Training curves ({args.split_mode} split)")
    print(f"Training done in {(time.time() - start) / 60:.1f} min | best val accuracy: {best_acc:.2%}")
    print(f"Model: {config.short(model_path)} | curves: {config.short(curves_path)}")


if __name__ == "__main__":
    main()
