"""Check how much the test set leaks from the training set, for each split mode.

For every test image we look for the most similar training image (pixel correlation of 64x64
thumbnails). A correlation above 0.99 means a near-copy, usually the neighbouring slice of the
same scan. We also report the accuracy of a "model" that just copies the label of that nearest
training image: if it is close to the real model's accuracy, the test set mostly rewards memory.

Usage:
    python check_leakage.py
"""
from collections import defaultdict

import numpy as np

from covid_ct import config
from covid_ct.similarity import load_thumbnails, normalize
from covid_ct.splits import get_split


def main():
    paths, labels, _ = get_split("slice")
    labels = np.array(labels)
    print(f"Loading {len(paths):,} images as small thumbnails...")
    md5s, thumbs = load_thumbnails([config.DATA_ROOT / p for p in paths])
    x = normalize(thumbs)

    copies = defaultdict(list)
    for i, h in enumerate(md5s):
        copies[h].append(i)

    print(f"\n{'Split mode':<12}{'near-copy in train':>20}{'1-NN baseline':>16}{'identical files':>18}")
    for mode in ("slice", "group"):
        try:
            _, _, split = get_split(mode)
        except FileNotFoundError as e:
            print(f"{mode:<12}skipped: {e}")
            continue
        train, test = np.array(split["train"]), np.array(split["test"])
        sims = x[test] @ x[train].T
        best = sims.argmax(axis=1)
        near_copy = (sims[np.arange(len(test)), best] > 0.99).mean()
        baseline = (labels[train[best]] == labels[test]).mean()

        in_train, in_test = set(train.tolist()), set(test.tolist())
        shared = sum(1 for g in copies.values()
                     if len(g) > 1 and in_train.intersection(g) and in_test.intersection(g))
        print(f"{mode:<12}{near_copy:>20.1%}{baseline:>16.2%}{shared:>18}")

    print("\nnear-copy in train: test images with a training image of pixel correlation > 0.99")
    print("1-NN baseline:      accuracy of copying the label of the most similar training image (no learning)")
    print("identical files:    byte-identical images with one copy in train and one in test")


if __name__ == "__main__":
    main()
