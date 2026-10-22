"""Paths, hyperparameters and device, shared by all scripts."""
from pathlib import Path

# Paths are relative to the project folder, so the scripts work from any directory
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"            # contains the NiCT/, nCT/ and pCT/ folders
OUTPUT_DIR = ROOT / "outputs"        # trained models and training histories
ASSETS_DIR = ROOT / "assets"         # figures used in the README

# Class names in the order ImageFolder assigns them (alphabetical, uppercase first)
CLASSES = ["NiCT", "nCT", "pCT"]
DESCRIPTIONS = {
    "NiCT": "non-informative, lungs not visible enough to judge",
    "nCT": "normal",
    "pCT": "COVID-19 positive",
}
POSITIVE_CLASS = "pCT"

# How the images are divided into train / validation / test
#   "slice": every slice is assigned at random (the original notebook split, leaks near-identical slices)
#   "group": near-identical slices of the same scan stay together on one side of the split
SPLIT_MODES = ("group", "slice")
DEFAULT_SPLIT_MODE = "group"
GROUP_SPLIT_PATH = ROOT / "splits" / "group_split.csv"
GROUP_THRESHOLD = 0.95               # pixel correlation above which two slices are linked

# Training settings (same values as in the notebook)
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
NUM_WORKERS = 4

# ImageNet statistics, expected by the pretrained network
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def model_path(mode):
    """File of the best model trained with a split mode ("slice" keeps the notebook's file name)."""
    return OUTPUT_DIR / ("best_model.pth" if mode == "slice" else f"best_model_{mode}.pth")


def history_path(mode):
    return OUTPUT_DIR / ("history.json" if mode == "slice" else f"history_{mode}.json")


def curves_path(mode):
    return ASSETS_DIR / ("training_curves.png" if mode == "slice" else f"training_curves_{mode}.png")


def short(path):
    """Path relative to the project folder, for display."""
    try:
        return Path(path).resolve().relative_to(ROOT)
    except ValueError:
        return Path(path)


def get_device():
    """Use the Apple GPU (MPS) or an NVIDIA GPU (CUDA) when available, otherwise the CPU."""
    import torch   # imported here so the scripts that only need numpy can run without PyTorch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
