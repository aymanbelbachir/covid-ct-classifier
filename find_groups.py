"""Group near-identical slices into "pseudo-patients" and split the groups into train / val / test.

The dataset has no patient IDs, but neighbouring slices of the same scan are almost identical.
Two slices are linked when their pixel correlation is above a threshold, and every chain of
linked slices becomes one group. A whole group then goes to a single split, so the test set
contains no near-copy of a training slice.

Usage:
    python find_groups.py                    # writes splits/group_split.csv
    python find_groups.py --threshold 0.97
"""
import argparse
import csv
import random
from collections import Counter

import numpy as np

from covid_ct import config
from covid_ct.similarity import load_thumbnails, nearest_neighbours, normalize
from covid_ct.splits import list_images


def connected_groups(nbr, sim, threshold):
    """Union-find: every chain of slices linked by a correlation above the threshold becomes one group."""
    parent = list(range(len(nbr)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(nbr)):
        for j, s in zip(nbr[i], sim[i]):
            if s > threshold:
                ra, rb = root(i), root(int(j))
                if ra != rb:
                    parent[ra] = rb
    roots = [root(i) for i in range(len(nbr))]
    ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}   # renumber groups 0, 1, 2, ...
    return np.array([ids[r] for r in roots])


def assign_groups(groups, labels, seed=config.SEED):
    """Give each group a split, stratified by the group's majority class (about 70 / 15 / 15 % of images)."""
    rng = random.Random(seed)
    n_groups = groups.max() + 1
    sizes = np.bincount(groups, minlength=n_groups)
    majority = [Counter(labels[groups == g]).most_common(1)[0][0] for g in range(n_groups)]

    split_of_group = {}
    for c in range(len(config.CLASSES)):
        gs = [g for g in range(n_groups) if majority[g] == c]
        rng.shuffle(gs)
        total, seen = sizes[gs].sum(), 0
        for g in gs:
            split_of_group[g] = "train" if seen < 0.70 * total else ("val" if seen < 0.85 * total else "test")
            seen += sizes[g]
    return [split_of_group[g] for g in groups]


def main():
    parser = argparse.ArgumentParser(description="Group near-identical slices and create a group-based split.")
    parser.add_argument("--threshold", type=float, default=config.GROUP_THRESHOLD,
                        help="pixel correlation above which two slices belong to the same group")
    args = parser.parse_args()

    paths, labels = list_images()
    labels = np.array(labels)
    print(f"Loading {len(paths):,} images as small thumbnails...")
    _, thumbs = load_thumbnails([config.DATA_ROOT / p for p in paths])

    print("Searching for near-identical slices...")
    nbr, sim = nearest_neighbours(normalize(thumbs, size=32))
    groups = connected_groups(nbr, sim, args.threshold)
    splits = assign_groups(groups, labels)

    config.GROUP_SPLIT_PATH.parent.mkdir(exist_ok=True)
    with open(config.GROUP_SPLIT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "class", "group", "split"])
        for path, label, group, split in zip(paths, labels, groups, splits):
            writer.writerow([path, config.CLASSES[label], group, split])

    sizes = np.bincount(groups)
    print(f"\n{len(sizes)} groups (threshold {args.threshold}), median size {int(np.median(sizes))}, largest {sizes.max()}")
    splits = np.array(splits)
    for name in ("train", "val", "test"):
        mask = splits == name
        per_class = ", ".join(f"{c} {np.sum(labels[mask] == i):,}" for i, c in enumerate(config.CLASSES))
        print(f"  {name:<5} {mask.sum():>6,} images in {len(np.unique(groups[mask])):>3} groups  ({per_class})")
    print(f"\nSaved {config.short(config.GROUP_SPLIT_PATH)}")


if __name__ == "__main__":
    main()
