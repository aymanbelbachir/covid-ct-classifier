"""Web demo: upload a chest CT slice, get the predicted class and a Grad-CAM heatmap.

Run it on your computer:
    python app.py          then open http://127.0.0.1:7860
Publish it on Hugging Face Spaces:
    python deploy_space.py
"""
import gradio as gr
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

from covid_ct import config
from covid_ct.data import eval_transform
from covid_ct.gradcam import GradCAM, to_display
from covid_ct.model import load_model

LABELS = {
    "NiCT": "Non-informative (lungs not visible enough)",
    "nCT": "Normal",
    "pCT": "COVID-19 positive",
}


def find_weights():
    """The model trained on the group split, or the one trained on the slice split."""
    for mode in config.SPLIT_MODES:
        path = config.model_path(mode)
        if path.exists():
            return path
    raise FileNotFoundError("No trained model in outputs/: run `python train.py` first")


device = torch.device("cpu")            # one image at a time is fast enough on the CPU
model = load_model(find_weights(), device)
cam = GradCAM(model)


def overlay(x, heatmap, alpha=0.4):
    """Blend the heatmap (blue = ignored, red = important) over the preprocessed image."""
    image = to_display(x)                                    # H x W x 3, values between 0 and 1
    colours = colormaps["jet"](heatmap)[..., :3]
    blended = (1 - alpha) * image + alpha * colours
    return Image.fromarray((blended * 255).astype(np.uint8))


def classify(image):
    """Return the probability of each class and the Grad-CAM image."""
    if image is None:
        return None, None
    x = eval_transform(image.convert("RGB")).unsqueeze(0).to(device)
    heatmap, probs, _ = cam(x)
    scores = {LABELS[c]: float(p) for c, p in zip(config.CLASSES, probs)}
    return scores, overlay(x, heatmap)


examples = sorted(str(p) for p in (config.ROOT / "examples").glob("*.jpg"))

with gr.Blocks(title="COVID-19 CT Scan Classifier") as demo:
    gr.Markdown(
        "# COVID-19 CT Scan Classifier\n"
        "Upload an axial chest CT slice. A MobileNetV3-Large fine-tuned on 19,685 CT slices classifies it as "
        "**normal**, **COVID-19 positive** or **non-informative**, and a Grad-CAM heatmap shows which regions "
        f"drove the prediction (red = most important). [Code on GitHub]({config.GITHUB_URL})"
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Chest CT slice")
            button = gr.Button("Classify", variant="primary")
        with gr.Column():
            label_out = gr.Label(num_top_classes=3, label="Prediction")
            heatmap_out = gr.Image(label="Grad-CAM: where the model looks")
    if examples:
        gr.Examples(examples=examples, inputs=image_in, outputs=[label_out, heatmap_out],
                    fn=classify, run_on_click=True, cache_examples=False)
    button.click(classify, inputs=image_in, outputs=[label_out, heatmap_out])
    gr.Markdown(
        "*This is a learning project, not a medical device. "
        "Do not use it to make decisions about anyone's health.*"
    )

if __name__ == "__main__":
    demo.launch()
