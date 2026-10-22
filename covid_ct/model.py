"""MobileNetV3-Large with a new 3-class head."""
import torch
from torch import nn
from torchvision import models

from . import config


def build_model(pretrained=True):
    """MobileNetV3-Large whose 1000-class ImageNet head is replaced by a 3-class one."""
    model = models.mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
    model.classifier = nn.Sequential(
        nn.Linear(960, 1280),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(1280, len(config.CLASSES)),
    )
    return model


def load_model(path, device):
    """Rebuild the architecture and load the trained weights from a file."""
    model = build_model(pretrained=False)   # no need to download the ImageNet weights here
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()
