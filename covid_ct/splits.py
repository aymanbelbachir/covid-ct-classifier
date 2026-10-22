"""Which images go to train, validation and test (no PyTorch needed)."""
import csv
import random

from . import config

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_images():
    """All images as (relative paths, labels), in the same order as torchvision's ImageFolder."""
    paths, labels = [], []
    for label, name in enumerate(config.CLASSES):
        folder = config.DATA_ROOT / name
        for fname in sorted(p.name for p in folder.iterdir() if p.is_file()):
            if fname.lower().endswith(IMG_EXTENSIONS):
                paths.append(f"{name}/{fname}")
                labels.append(label)
    return paths, labels


def slice_split(labels, seed=config.SEED):
    """Original notebook split: the slices of each class are shuffled and cut into 70 / 15 / 15 %."""
    rng = random.Random(seed)
    by_class = {i: [] for i in range(len(config.CLASSES))}
    for idx, label in enumerate(labels):
        by_class[label].append(idx)

    splits = {"train": [], "val": [], "test": []}
    for idxs in by_class.values():
        rng.shuffle(idxs)
        n_train = int(0.70 * len(idxs))
        n_val = int(0.15 * len(idxs))
        splits["train"] += idxs[:n_train]
        splits["val"] += idxs[n_train:n_train + n_val]
        splits["test"] += idxs[n_train + n_val:]
    return splits


def group_split(paths):
    """Split stored in splits/group_split.csv (created by find_groups.py)."""
    if not config.GROUP_SPLIT_PATH.exists():
        raise FileNotFoundError(f"{config.short(config.GROUP_SPLIT_PATH)} not found: run `python find_groups.py` first")
    with open(config.GROUP_SPLIT_PATH, newline="") as f:
        assigned = {row["path"]: row["split"] for row in csv.DictReader(f)}

    missing = [p for p in paths if p not in assigned]
    if missing:
        raise ValueError(f"{len(missing)} images are not in {config.short(config.GROUP_SPLIT_PATH)}, "
                         f"e.g. {missing[0]}: run `python find_groups.py` again")

    splits = {"train": [], "val": [], "test": []}
    for idx, path in enumerate(paths):
        splits[assigned[path]].append(idx)
    return splits


def get_split(mode):
    """Return (relative paths, labels, {split name: list of image indices}) for a split mode."""
    paths, labels = list_images()
    if mode == "slice":
        return paths, labels, slice_split(labels)
    if mode == "group":
        return paths, labels, group_split(paths)
    raise ValueError(f"Unknown split mode {mode!r}, expected one of {config.SPLIT_MODES}")
