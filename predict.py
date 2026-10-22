"""Predict the class of one or more CT slices with the trained model.

Usage:
    python predict.py data/pCT/pCT3689.jpg
    python predict.py image1.jpg image2.jpg
"""
import argparse

import torch
from PIL import Image

from covid_ct import config
from covid_ct.data import eval_transform
from covid_ct.model import load_model


def main():
    parser = argparse.ArgumentParser(description="Classify CT slices as NiCT, nCT or pCT.")
    parser.add_argument("images", nargs="+", help="path(s) to the image file(s)")
    parser.add_argument("--weights", default=str(config.MODEL_PATH), help="path to the trained model")
    args = parser.parse_args()

    device = config.get_device()
    model = load_model(args.weights, device)

    for path in args.images:
        image = Image.open(path).convert("RGB")
        x = eval_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu()

        best = probs.argmax().item()
        name = config.CLASSES[best]
        print(f"{path}: {name} ({config.DESCRIPTIONS[name]}), {probs[best].item():.2%}")
        for c, p in zip(config.CLASSES, probs.tolist()):
            print(f"    {c:<5} {p:7.2%}")


if __name__ == "__main__":
    main()
