"""Confusion matrix, precision, recall, F1, sensitivity and specificity."""
import math

import torch


def confusion_matrix(true, preds, n_classes):
    """Rows = true class, columns = predicted class."""
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    for t, p in zip(true.tolist(), preds.tolist()):
        cm[t, p] += 1
    return cm


def wilson_interval(k, n, z=1.96):
    """95% confidence interval for a proportion k / n (Wilson score interval)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def per_class_report(cm, classes):
    """Precision, recall, F1 and support of each class.

    precision = of the images predicted as this class, how many really are
    recall    = of the images of this class, how many were found
    """
    rows = []
    for i, name in enumerate(classes):
        tp = cm[i, i].item()
        support = cm[i].sum().item()        # true images of this class
        predicted = cm[:, i].sum().item()   # images predicted as this class
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "class": name, "precision": precision, "recall": recall, "f1": f1,
            "support": support, "recall_ci": wilson_interval(tp, support),
        })
    return rows


def binary_report(cm, positive):
    """Sensitivity and specificity of 'positive class' versus all other classes."""
    tp = cm[positive, positive].item()
    fn = cm[positive].sum().item() - tp
    fp = cm[:, positive].sum().item() - tp
    tn = cm.sum().item() - tp - fn - fp
    return {
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0, "sensitivity_ci": wilson_interval(tp, tp + fn),
        "specificity": tn / (tn + fp) if tn + fp else 0.0, "specificity_ci": wilson_interval(tn, tn + fp),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }
