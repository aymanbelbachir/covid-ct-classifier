"""Evaluate a trained model: accuracy, precision, recall, F1, sensitivity, specificity.

Usage:
    python evaluate.py                        # model trained on the group split, test set
    python evaluate.py --split-mode slice     # model trained on the original notebook split
    python evaluate.py --split val            # validation set
"""
import argparse
import sys
from pathlib import Path

from covid_ct import config
from covid_ct.data import get_dataloaders
from covid_ct.engine import get_predictions
from covid_ct.metrics import binary_report, confusion_matrix, per_class_report, wilson_interval
from covid_ct.model import load_model

NOTES = {
    "slice": "Note: with the slice split, near-identical slices of the same scan appear in both the training\n"
             "and the test set, so these numbers are optimistic for new patients (see check_leakage.py).",
    "group": "Note: with the group split, near-identical slices stay on one side of the split. The dataset has no\n"
             "patient IDs, so distant slices of the same patient could still fall on both sides.",
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained COVID-19 CT classifier.")
    parser.add_argument("--split-mode", choices=config.SPLIT_MODES, default=config.DEFAULT_SPLIT_MODE,
                        help="split the model was trained with")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="data split to evaluate")
    parser.add_argument("--weights", default=None, help="path to the trained model (default: the split mode's model)")
    args = parser.parse_args()

    weights = Path(args.weights) if args.weights else config.model_path(args.split_mode)
    if not weights.exists():
        sys.exit(f"{config.short(weights)} not found: train it first with "
                 f"`python train.py --split-mode {args.split_mode}`")

    device = config.get_device()
    loader = get_dataloaders([args.split], mode=args.split_mode)[args.split]
    model = load_model(weights, device)
    print(f"Evaluating {config.short(weights)} on the {args.split} set of the {args.split_mode} split "
          f"({len(loader.dataset):,} images, device: {device})\n")

    preds, true = get_predictions(model, loader, device)
    classes = config.CLASSES
    cm = confusion_matrix(true, preds, len(classes))

    n, correct = cm.sum().item(), cm.trace().item()
    lo, hi = wilson_interval(correct, n)
    print(f"Accuracy: {correct / n:.2%}  (95% CI {lo:.2%} - {hi:.2%}) | {n - correct} errors\n")

    rows = per_class_report(cm, classes)
    print(f"{'Class':<10}{'Precision':>10}{'Recall':>9}{'F1':>8}{'Support':>9}   Recall 95% CI")
    for r in rows:
        lo, hi = r["recall_ci"]
        print(f"{r['class']:<10}{r['precision']:>10.2%}{r['recall']:>9.2%}{r['f1']:>8.4f}"
              f"{r['support']:>9,}   {lo:.1%} - {hi:.1%}")
    macro = {k: sum(r[k] for r in rows) / len(rows) for k in ("precision", "recall", "f1")}
    print(f"{'Macro avg':<10}{macro['precision']:>10.2%}{macro['recall']:>9.2%}{macro['f1']:>8.4f}\n")

    b = binary_report(cm, classes.index(config.POSITIVE_CLASS))
    print(f"COVID-19 ({config.POSITIVE_CLASS}) versus the other classes:")
    lo, hi = b["sensitivity_ci"]
    print(f"  Sensitivity: {b['sensitivity']:.2%}  (95% CI {lo:.1%} - {hi:.1%}) | "
          f"{b['fn']} of {b['tp'] + b['fn']:,} positive slices missed")
    lo, hi = b["specificity_ci"]
    print(f"  Specificity: {b['specificity']:.2%}  (95% CI {lo:.2%} - {hi:.2%}) | "
          f"{b['fp']} false alarm{'s' if b['fp'] != 1 else ''} on {b['tn'] + b['fp']:,} other slices\n")

    print("Confusion matrix (rows = true class, columns = predicted class):")
    print("      " + "".join(f"{c:>7}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:<6}" + "".join(f"{cm[i, j].item():>7}" for j in range(len(classes))))

    print("\n" + NOTES[args.split_mode])


if __name__ == "__main__":
    main()
