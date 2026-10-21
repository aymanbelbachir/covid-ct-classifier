# COVID-19 CT Scan Classifier

A PyTorch image classifier that sorts chest CT slices into three classes: normal, COVID-19 positive and non-informative. It uses transfer learning on MobileNetV3-Large and reaches **99.59% accuracy** on a held-out test set of 2,956 slices.

> This is a learning project. It is not a medical device and must not be used to make decisions about anyone's health.

## The task

Each image is a single slice from a chest CT scan. The model assigns it to one of three classes:

| Class | Meaning | Images |
|---|---|---:|
| `nCT` | Normal: the lungs are visible and healthy | 9,979 |
| `pCT` | Positive: the lungs show signs of COVID-19 pneumonia | 4,001 |
| `NiCT` | Non-informative: the slice does not show enough of the lungs to judge | 5,705 |

## Results

| Split | Images | Accuracy |
|---|---:|---:|
| Validation | 2,951 | 99.53% |
| **Test** | **2,956** | **99.59%** |

Confusion matrix on the test set (rows = true class, columns = predicted class):

| | NiCT | nCT | pCT |
|---|---:|---:|---:|
| **NiCT** | 854 | 3 | 0 |
| **nCT** | 2 | 1,495 | 1 |
| **pCT** | 5 | 1 | 595 |

Ten of the 12 errors involve the non-informative class: slices at the edge of the lungs, where it is ambiguous whether enough lung is visible to judge. One COVID-positive slice was classified as normal, which is the most serious type of error for this task.

![Training curves](assets/training_curves.png)

## How it works

- **Data:** the slices are split per class into 70% training, 15% validation and 15% test, with a fixed seed (42). Training images are augmented with random resized crops and horizontal flips. All images are normalized with the ImageNet statistics.
- **Model:** MobileNetV3-Large pretrained on ImageNet. Its original 1,000-class head is replaced by a new one: 960 → 1,280 (ReLU, 20% dropout) → 3. The whole network is fine-tuned (4.2M trainable parameters).
- **Training:** cross-entropy loss, AdamW (learning rate 1e-4, weight decay 1e-4), batch size 16, 25 epochs. The model with the best validation accuracy is kept.
- **Hardware:** Apple Silicon MacBook, PyTorch MPS backend. Training took about 64 minutes.

## Getting started

```bash
git clone https://github.com/aymanbelbachir/covid-ct-classifier.git
cd covid-ct-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/azaemon/preprocessed-ct-scans-for-covid19) and move the `nCT`, `NiCT` and `pCT` folders from `Original CT Scans` into a `data/` folder:

```
data/
├── NiCT/
├── nCT/
└── pCT/
```

Then open `classifier.ipynb` with `jupyter notebook` and run all cells. The trained model is saved to `outputs/best_model.pth`.

## Limitations

- **The split is per slice, not per patient.** Neighbouring slices of the same patient look very similar and can end up in both the training and the test set, so the test accuracy is likely higher than it would be on new patients.
- **One data source.** All scans come from two hospitals in Wuhan. The model has not been tested on other scanners or populations.
- **Single slices only.** A real diagnosis looks at the whole scan together with symptoms and lab results.

## Acknowledgements

- Dataset: [CT Scans for COVID-19 Classification](https://www.kaggle.com/datasets/azaemon/preprocessed-ct-scans-for-covid19) (CC BY 4.0), built from the iCTCF data of Ning et al. (2020).
- Model: Howard et al., "Searching for MobileNetV3", ICCV 2019. Pretrained weights from torchvision.
