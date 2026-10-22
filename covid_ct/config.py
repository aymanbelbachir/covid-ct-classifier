"""Paths, hyperparameters and device, shared by all scripts."""
from pathlib import Path

import torch

# Paths are relative to the project folder, so the scripts work from any directory
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"            # contains the NiCT/, nCT/ and pCT/ folders
OUTPUT_DIR = ROOT / "outputs"        # trained model and training history
ASSETS_DIR = ROOT / "assets"         # figures used in the README
MODEL_PATH = OUTPUT_DIR / "best_model.pth"
HISTORY_PATH = OUTPUT_DIR / "history.json"

# Class names in the order ImageFolder assigns them (alphabetical, uppercase first)
CLASSES = ["NiCT", "nCT", "pCT"]
DESCRIPTIONS = {
    "NiCT": "non-informative, lungs not visible enough to judge",
    "nCT": "normal",
    "pCT": "COVID-19 positive",
}
POSITIVE_CLASS = "pCT"

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


def short(path):
    """Path relative to the project folder, for display."""
    try:
        return Path(path).resolve().relative_to(ROOT)
    except ValueError:
        return Path(path)


def get_device():
    """Use the Apple GPU (MPS) or an NVIDIA GPU (CUDA) when available, otherwise the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
