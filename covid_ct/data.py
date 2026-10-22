"""Image transforms, the train/validation/test split and the data loaders."""
import random

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from . import config

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(config.IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(config.MEAN, config.STD),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(config.MEAN, config.STD),
])


def split_indices(targets, n_classes, seed=config.SEED):
    """Split the images of each class into 70% train, 15% validation and 15% test.

    Same procedure and seed as the notebook, so the split is identical.
    """
    rng = random.Random(seed)
    by_class = {i: [] for i in range(n_classes)}
    for idx, label in enumerate(targets):
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


def get_dataloaders(split_names=("train", "val", "test")):
    """Return a dict {split name: DataLoader} for the requested splits."""
    base = datasets.ImageFolder(config.DATA_ROOT)
    if base.classes != config.CLASSES:
        raise ValueError(f"Expected the folders {config.CLASSES} in {config.DATA_ROOT}, found {base.classes}")

    indices = split_indices(base.targets, len(base.classes))
    loaders = {}
    for name in split_names:
        transform = train_transform if name == "train" else eval_transform
        dataset = Subset(datasets.ImageFolder(config.DATA_ROOT, transform=transform), indices[name])
        loaders[name] = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=(name == "train"),
            num_workers=config.NUM_WORKERS,
            persistent_workers=config.NUM_WORKERS > 0,
        )
    return loaders
