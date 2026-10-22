"""Grad-CAM: which parts of an image made the model choose a class.

The last convolutional block of MobileNetV3 turns a 224x224 image into 960 feature maps of 7x7.
Grad-CAM weights each map by how much it pushes the score of the chosen class up (the average
of its gradient), adds them up and keeps the positive part. The result is a 7x7 heatmap that is
enlarged to the size of the image: warm areas are the ones the decision relied on.
"""
import torch
import torch.nn.functional as F

from . import config


class GradCAM:
    def __init__(self, model):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        model.features[-1].register_forward_hook(self._save_activations)

    def _save_activations(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradients)       # called during backward()

    def _save_gradients(self, grad):
        self.gradients = grad

    def __call__(self, x, target=None):
        """x: one preprocessed image [1, 3, H, W].

        Returns (heatmap as an H x W array between 0 and 1, class probabilities, explained class).
        """
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            if target is None:
                target = probs.argmax().item()           # explain the predicted class by default
            logits[0, target].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)            # importance of each feature map
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # weighted sum, positive part
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), probs.detach().cpu(), target


def to_display(x):
    """Undo the normalization of a preprocessed image [1, 3, H, W] so it can be shown with matplotlib."""
    mean = torch.tensor(config.MEAN).view(3, 1, 1)
    std = torch.tensor(config.STD).view(3, 1, 1)
    img = x[0].detach().cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()
