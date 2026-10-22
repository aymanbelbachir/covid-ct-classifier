"""Export the trained model to ONNX, for the demo that runs in the browser (web/).

The exported model takes one image with values between 0 and 1, shape [1, 3, 224, 224], and returns:
    probs  [1, 3]              probability of each class (NiCT, nCT, pCT)
    cams   [1, 3, 224, 224]    Grad-CAM heatmap of each class, between 0 and 1

Grad-CAM needs the gradient of a class score with respect to the last feature maps. The layers
after those maps are simple (average pooling, Linear, ReLU, Linear), so this gradient can be
written as a formula. The heatmaps are therefore computed inside the exported model and the
browser does not need backpropagation.

Usage:
    python export_onnx.py          # outputs/best_model_group.pth -> outputs/model.onnx
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms

from covid_ct import config
from covid_ct.data import eval_transform
from covid_ct.gradcam import GradCAM
from covid_ct.model import load_model

ONNX_PATH = config.OUTPUT_DIR / "model.onnx"


class BrowserModel(nn.Module):
    """The trained network plus normalization and Grad-CAM, as a single forward pass."""

    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.fc1 = model.classifier[0]        # Linear 960 -> 1280
        self.fc2 = model.classifier[3]        # Linear 1280 -> 3 (ReLU and Dropout in between)
        self.register_buffer("mean", torch.tensor(config.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(config.STD).view(1, 3, 1, 1))

    def forward(self, image):
        x = (image - self.mean) / self.std
        acts = self.features(x)                               # [1, 960, 7, 7]
        n, k, h, w = acts.shape
        pooled = acts.mean(dim=(2, 3))                        # [1, 960]
        z1 = self.fc1(pooled)                                 # [1, 1280]
        logits = self.fc2(torch.relu(z1))                     # [1, 3]
        probs = torch.softmax(logits, dim=1)

        # d logit_c / d pooled = (W2[c] * relu'(z1)) @ W1; average pooling spreads it over the h*w positions
        relu_grad = (z1 > 0).to(z1.dtype)
        weights = (self.fc2.weight * relu_grad) @ self.fc1.weight / (h * w)    # [3, 960]
        cams = torch.relu(weights @ acts.reshape(k, h * w))   # [3, 49]
        cams = cams.reshape(1, -1, h, w)
        cams = F.interpolate(cams, size=image.shape[-2:], mode="bilinear", align_corners=False)
        low = cams.amin(dim=(2, 3), keepdim=True)
        high = cams.amax(dim=(2, 3), keepdim=True)
        cams = (cams - low) / (high - low + 1e-8)
        return probs, cams


def check(onnx_path, model, images):
    """Compare the ONNX model with PyTorch + Grad-CAM on a few images."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    to_unit = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(config.IMG_SIZE),
                                  transforms.ToTensor()])
    cam = GradCAM(model)
    worst_prob, worst_cam = 0.0, 0.0
    for path in images:
        image = Image.open(path).convert("RGB")
        probs, cams = session.run(None, {"image": to_unit(image).unsqueeze(0).numpy()})
        heatmap, ref_probs, target = cam(eval_transform(image).unsqueeze(0))
        worst_prob = max(worst_prob, float(np.abs(probs[0] - ref_probs.numpy()).max()))
        worst_cam = max(worst_cam, float(np.abs(cams[0, target] - heatmap).max()))
        print(f"  {Path(path).name}: {config.CLASSES[int(probs[0].argmax())]} {probs[0].max():.2%}")
    print(f"Largest difference with PyTorch: probabilities {worst_prob:.1e}, heatmaps {worst_cam:.1e}")


def main():
    parser = argparse.ArgumentParser(description="Export the trained model to ONNX for the browser demo.")
    parser.add_argument("--weights", default=str(config.model_path("group")), help="trained model")
    parser.add_argument("--out", default=str(ONNX_PATH), help="ONNX file to write")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"{config.short(args.weights)} not found: train the model first with `python train.py`")

    model = load_model(args.weights, torch.device("cpu"))
    browser_model = BrowserModel(model).eval()
    dummy = torch.rand(1, 3, config.IMG_SIZE, config.IMG_SIZE)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(browser_model, (dummy,), args.out, input_names=["image"],
                      output_names=["probs", "cams"], opset_version=17, dynamo=False)
    size = Path(args.out).stat().st_size
    print(f"Saved {config.short(args.out)} ({size / 1e6:.1f} MB)")

    examples = sorted((config.ROOT / "examples").glob("*.jpg"))
    try:
        check(args.out, model, examples)
    except ImportError:
        print("Install onnxruntime (pip install onnxruntime) to check the exported model against PyTorch.")


if __name__ == "__main__":
    main()
