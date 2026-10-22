"""Publish the in-browser demo (web/) on Hugging Face Spaces, as a free static Space.

The page runs the ONNX model in the visitor's browser, so no server is needed.

Usage:
    python export_onnx.py              # once, creates outputs/model.onnx
    python deploy_space.py --preview   # try the page on this computer: http://127.0.0.1:8000
    hf auth login                      # once, connects this computer to your Hugging Face account
    python deploy_space.py             # creates or updates https://huggingface.co/spaces/<your-name>/covid-ct-classifier
"""
import argparse
import functools
import http.server
import json
import shutil
import tempfile
import webbrowser
from pathlib import Path

from covid_ct import config

SPACE_NAME = "covid-ct-classifier"
WEB_DIR = config.ROOT / "web"
ONNX_PATH = config.OUTPUT_DIR / "model.onnx"
PREVIEW_DIR = config.OUTPUT_DIR / "space_preview"

CARD = """---
title: COVID-19 CT Classifier
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
short_description: Chest CT classifier with Grad-CAM, runs in your browser
---

# COVID-19 CT Classifier

Drop a chest CT slice to classify it as normal, COVID-19 positive or non-informative, and see a
Grad-CAM heatmap of the regions that drove the prediction.

The model is a MobileNetV3-Large fine-tuned on 19,685 CT slices, exported to ONNX and executed in
the browser with ONNX Runtime Web: images never leave the visitor's device. On 29 CT scans held
out from training it reaches 96.9% accuracy. Code, training and evaluation: {github}

The example images come from the [CT Scans for COVID-19 Classification](https://www.kaggle.com/datasets/azaemon/preprocessed-ct-scans-for-covid19)
dataset (CC BY 4.0), built from the iCTCF data of Ning et al. (2020).

Designed and built by [Ayman Belbachir](https://github.com/aymanbelbachir).

This is a learning project, not a medical device. Do not use it to make decisions about anyone's health.
"""


def build(folder, onnx_path, repo_id=None):
    """Copy the page, the model and the examples into a folder, ready to be served."""
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    shutil.copytree(WEB_DIR, folder)
    shutil.copytree(config.ROOT / "examples", folder / "examples")
    shutil.copy(onnx_path, folder / "model.onnx")

    meta = {"file": "model.onnx", "bytes": Path(onnx_path).stat().st_size}
    if repo_id:   # direct link to the file, used by the page if the static host does not serve it
        meta["fallback"] = f"https://huggingface.co/spaces/{repo_id}/resolve/main/model.onnx"
    (folder / "model.json").write_text(json.dumps(meta))
    (folder / "README.md").write_text(CARD.format(github=config.GITHUB_URL))


def preview(onnx_path, port):
    """Serve the page on this computer."""
    build(PREVIEW_DIR, onnx_path)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PREVIEW_DIR))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        url = f"http://127.0.0.1:{port}"
        print(f"Preview running at {url} (press Ctrl + C to stop)")
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Publish the in-browser demo on Hugging Face Spaces.")
    parser.add_argument("--repo", default=None, help="Space id, e.g. your-name/covid-ct-classifier")
    parser.add_argument("--onnx", default=str(ONNX_PATH), help="exported model (see export_onnx.py)")
    parser.add_argument("--preview", action="store_true", help="serve the page on this computer instead")
    parser.add_argument("--port", type=int, default=8000, help="port of the preview server")
    parser.add_argument("--dry-run", metavar="FOLDER", default=None, help="build the Space folder here, do not upload")
    args = parser.parse_args()

    if not Path(args.onnx).exists():
        raise SystemExit(f"{config.short(args.onnx)} not found: export the model first with `python export_onnx.py`")

    if args.preview:
        return preview(args.onnx, args.port)

    if args.dry_run:
        build(args.dry_run, args.onnx, args.repo)
        for p in sorted(Path(args.dry_run).rglob("*")):
            if p.is_file():
                print(p.relative_to(args.dry_run))
        return

    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = args.repo or f"{api.whoami()['name']}/{SPACE_NAME}"
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "space"
        build(folder, args.onnx, repo_id)
        api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=folder, repo_type="space",
                          commit_message="Update the demo")
    print(f"Done: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
