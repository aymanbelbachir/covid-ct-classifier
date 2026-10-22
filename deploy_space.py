"""Publish the web demo (app.py) on Hugging Face Spaces.

Usage:
    hf auth login                 # once: connects this computer to your Hugging Face account
    python deploy_space.py        # creates or updates https://huggingface.co/spaces/<your-name>/covid-ct-classifier
    python deploy_space.py --dry-run space_preview   # only builds the Space folder locally, to inspect it
"""
import argparse
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

from covid_ct import config

SPACE_NAME = "covid-ct-classifier"
CODE_FILES = [
    "app.py",
    "covid_ct/__init__.py", "covid_ct/config.py", "covid_ct/splits.py",
    "covid_ct/data.py", "covid_ct/model.py", "covid_ct/gradcam.py",
]

# Gradio itself is installed by Hugging Face (sdk_version below); the CPU build of PyTorch is much smaller
REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
numpy
matplotlib
pillow
"""

CARD = """---
title: COVID-19 CT Scan Classifier
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: {gradio_version}
app_file: app.py
pinned: false
short_description: Classifies chest CT slices with Grad-CAM heatmaps
---

# COVID-19 CT Scan Classifier

Upload an axial chest CT slice to classify it as normal, COVID-19 positive or non-informative.
A Grad-CAM heatmap shows which regions of the image drove the prediction.

The model is a MobileNetV3-Large fine-tuned on 19,685 CT slices. It reaches 96.9% accuracy on
29 CT scans held out from training. Code, training and evaluation: {github}

The example images come from the [CT Scans for COVID-19 Classification](https://www.kaggle.com/datasets/azaemon/preprocessed-ct-scans-for-covid19)
dataset (CC BY 4.0), built from the iCTCF data of Ning et al. (2020).

This is a learning project, not a medical device. Do not use it to make decisions about anyone's health.
"""


def build(folder, weights):
    """Copy everything the Space needs into a folder."""
    folder = Path(folder)
    for name in CODE_FILES:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(config.ROOT / name, folder / name)

    target = folder / config.model_path("group").relative_to(config.ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(weights, target)

    (folder / "examples").mkdir(exist_ok=True)
    for image in sorted((config.ROOT / "examples").glob("*.jpg")):
        shutil.copy(image, folder / "examples" / image.name)

    (folder / "requirements.txt").write_text(REQUIREMENTS)
    (folder / "README.md").write_text(CARD.format(gradio_version=version("gradio"), github=config.GITHUB_URL))


def main():
    parser = argparse.ArgumentParser(description="Publish the demo on Hugging Face Spaces.")
    parser.add_argument("--repo", default=None, help="Space id, e.g. your-name/covid-ct-classifier")
    parser.add_argument("--weights", default=str(config.model_path("group")), help="trained model to publish")
    parser.add_argument("--dry-run", metavar="FOLDER", default=None, help="build the Space folder here, do not upload")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"{config.short(args.weights)} not found: train the model first with `python train.py`")

    if args.dry_run:
        build(args.dry_run, args.weights)
        for p in sorted(Path(args.dry_run).rglob("*")):
            if p.is_file():
                print(p.relative_to(args.dry_run))
        return

    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = args.repo or f"{api.whoami()['name']}/{SPACE_NAME}"
    with tempfile.TemporaryDirectory() as tmp:
        build(tmp, args.weights)
        api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=tmp, repo_type="space",
                          commit_message="Update the demo")
    print(f"Done: https://huggingface.co/spaces/{repo_id}")
    print("The Space takes a few minutes to build the first time.")


if __name__ == "__main__":
    main()
