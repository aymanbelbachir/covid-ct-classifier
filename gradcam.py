"""Draw Grad-CAM heatmaps: where the model looks when it classifies a slice.

Usage:
    python gradcam.py data/pCT/pCT3689.jpg data/nCT/nCT7057.jpg   # your own images
    python gradcam.py --examples 2                                # 2 random test images per class
    python gradcam.py --errors                                    # misclassified test images, missed COVID first
The figure is saved to assets/gradcam.png (change it with --out).
"""
import argparse
import math
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from covid_ct import config
from covid_ct.data import eval_transform, get_dataloaders
from covid_ct.engine import get_predictions
from covid_ct.gradcam import GradCAM, to_display
from covid_ct.model import load_model
from covid_ct.splits import get_split, load_groups


def pick_images(args, model, device):
    """Return a list of (path, true class or None) to explain."""
    items = []
    for p in args.images:
        parent = Path(p).parent.name
        items.append((Path(p), parent if parent in config.CLASSES else None))

    if args.examples or args.errors:
        paths, labels, split = get_split(args.split_mode)
        test = split["test"]
        if args.examples:
            rng = random.Random(config.SEED)
            for c in range(len(config.CLASSES)):
                of_class = [i for i in test if labels[i] == c]
                for i in rng.sample(of_class, min(args.examples, len(of_class))):
                    items.append((config.DATA_ROOT / paths[i], config.CLASSES[c]))
        if args.errors:
            loader = get_dataloaders(["test"], mode=args.split_mode)["test"]
            preds, true = get_predictions(model, loader, device)
            preds, true = preds.tolist(), true.tolist()
            pos = config.CLASSES.index(config.POSITIVE_CLASS)
            wrong = [k for k in range(len(true)) if preds[k] != true[k]]
            # most serious errors first: missed COVID, then false alarms, then the rest
            wrong.sort(key=lambda k: 0 if true[k] == pos else (1 if preds[k] == pos else 2))

            groups = load_groups() if args.split_mode == "group" else {}
            per_group, picked = Counter(), []
            for k in wrong:
                g = groups.get(paths[test[k]])     # the test loader keeps the order of split["test"]
                if g is not None and per_group[g] >= 2:
                    continue                       # at most 2 errors per scan, to show different scans
                per_group[g] += 1
                picked.append(k)
                if len(picked) == args.max_errors:
                    break
            print(f"{len(wrong)} misclassified test images, showing {len(picked)} (missed COVID first)")
            for k in picked:
                i = test[k]
                items.append((config.DATA_ROOT / paths[i], config.CLASSES[labels[i]]))
    return items


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM heatmaps for the COVID-19 CT classifier.")
    parser.add_argument("images", nargs="*", help="image files to explain")
    parser.add_argument("--examples", type=int, default=0, help="number of random test images per class")
    parser.add_argument("--errors", action="store_true", help="add misclassified test images (missed COVID first)")
    parser.add_argument("--max-errors", type=int, default=5, help="maximum number of errors to show")
    parser.add_argument("--split-mode", choices=config.SPLIT_MODES, default=config.DEFAULT_SPLIT_MODE,
                        help="split whose test set and model are used")
    parser.add_argument("--weights", default=None, help="path to the trained model (default: the split mode's model)")
    parser.add_argument("--out", default=str(config.ASSETS_DIR / "gradcam.png"), help="output figure")
    args = parser.parse_args()

    weights = Path(args.weights) if args.weights else config.model_path(args.split_mode)
    if not weights.exists():
        raise SystemExit(f"{config.short(weights)} not found: train it first with "
                         f"`python train.py --split-mode {args.split_mode}`")

    device = config.get_device()
    model = load_model(weights, device)
    items = pick_images(args, model, device)
    if not items:
        parser.error("give image paths, --examples N or --errors")

    cam = GradCAM(model)
    cols = min(4, len(items))
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.4 * rows), squeeze=False)

    for ax, (path, true) in zip(axes.flat, items):
        x = eval_transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        heatmap, probs, pred = cam(x)
        name = config.CLASSES[pred]
        ax.imshow(to_display(x))
        ax.imshow(heatmap, cmap="jet", alpha=0.35)
        title = f"{path.name}\npredicted {name} ({probs[pred].item():.0%})"
        if true is not None:
            title += f" | true {true}" + ("" if true == name else "  ✗")
        ax.set_title(title, fontsize=10, color="black" if true in (None, name) else "red")
        ax.axis("off")
        print(f"{path}: predicted {name} ({probs[pred].item():.2%})" + (f", true {true}" if true else ""))

    for ax in axes.flat[len(items):]:
        ax.axis("off")

    plt.tight_layout()
    Path(args.out).parent.mkdir(exist_ok=True)
    plt.savefig(args.out, dpi=120)
    print(f"\nSaved {config.short(args.out)}")


if __name__ == "__main__":
    main()
