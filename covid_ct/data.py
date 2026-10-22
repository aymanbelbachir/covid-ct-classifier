"""Image transforms and data loaders."""
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from . import config
from .splits import get_split

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


def get_dataloaders(split_names=("train", "val", "test"), mode=config.DEFAULT_SPLIT_MODE):
    """Return a dict {split name: DataLoader} for the requested splits and split mode."""
    base = datasets.ImageFolder(config.DATA_ROOT)
    if base.classes != config.CLASSES:
        raise ValueError(f"Expected the folders {config.CLASSES} in {config.DATA_ROOT}, found {base.classes}")

    paths, _, indices = get_split(mode)
    folder_paths = [Path(p).relative_to(config.DATA_ROOT).as_posix() for p, _ in base.samples]
    if folder_paths != paths:
        raise RuntimeError("The image list does not match ImageFolder's order")

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
