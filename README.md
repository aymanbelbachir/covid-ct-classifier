# COVID-19 CT Scan Classifier

A PyTorch image classifier that sorts chest CT slices into three classes: normal, COVID-19 positive and non-informative. It uses transfer learning on MobileNetV3-Large. Evaluated on scans it has never seen, it reaches **96.9% accuracy** and detects **90% of COVID-positive slices**.

The project also shows that the usual way of splitting this dataset, slice by slice, leaks near-identical images from the training set into the test set and inflates the accuracy to 99.6%, and how to fix it.

> This is a learning project. It is not a medical device and must not be used to make decisions about anyone's health.

## The task

Each image is a single slice from a chest CT scan. The model assigns it to one of three classes:

| Class | Meaning | Images |
|---|---|---:|
| `nCT` | Normal: the lungs are visible and healthy | 9,979 |
| `pCT` | Positive: the lungs show signs of COVID-19 pneumonia | 4,001 |
| `NiCT` | Non-informative: the slice does not show enough of the lungs to judge | 5,705 |

## Results

### An honest split: whole scans, not slices

The dataset has no patient IDs, and neighbouring slices of a scan are nearly identical. A random split slice by slice therefore puts near-copies of training slices in the test set. `find_groups.py` links every pair of slices whose pixel correlation is above 0.95 and turns each chain of linked slices into a group. This gives 189 groups, a proxy for scans, with a median of 81 slices each. Whole groups are then assigned to the training (137 groups), validation (23) or test set (29).

| | Per-slice split | Per-scan split |
|---|---:|---:|
| Test slices with a near-copy in the training set | 97.1% | 0.0% |
| Nearest-neighbour baseline (no learning) | 99.66% | 84.07% |
| **Model test accuracy** | 99.59% | **96.88%** |
| **COVID-19 sensitivity** | 99.00% | **89.96%** |
| **COVID-19 specificity** | 99.96% | **99.50%** |
| Errors on the test set | 12 of 2,956 | 80 of 2,567 |

The nearest-neighbour baseline copies the label of the most similar training slice. With the per-slice split it already scores 99.66%, so the model's 99.59% said very little. With the per-scan split the model beats that baseline by 13 points, so it did learn to read the images, but it misses 10% of COVID-positive slices instead of 1%. `check_leakage.py` reproduces the first two rows.

### Detailed results (per-scan split)

| Class | Precision | Recall | F1 | Recall 95% CI |
|---|---:|---:|---:|---:|
| NiCT | 96.97% | 99.61% | 0.983 | 98.9–99.9% |
| nCT | 96.38% | 98.24% | 0.973 | 97.3–98.8% |
| pCT | 98.01% | 89.96% | 0.938 | 87.2–92.2% |

Confusion matrix on the test set (rows = true class, columns = predicted class):

| | NiCT | nCT | pCT |
|---|---:|---:|---:|
| **NiCT** | 769 | 2 | 1 |
| **nCT** | 13 | 1,225 | 9 |
| **pCT** | 11 | 44 | 493 |

The most frequent error is also the most serious one: 44 COVID-positive slices predicted as normal. The errors are concentrated in a few scans: the 80 errors come from 10 of the 29 test scans, and three scans account for 53 of them. At the scan level, all 8 test scans that contain COVID-positive slices have at least one slice flagged as positive, and 3 of the 21 other scans have at least one false alarm. With only 8 positive scans in the test set, the per-slice confidence intervals understate the real uncertainty.

![Training curves on the per-scan split](assets/training_curves_group.png)

Validation accuracy levels off around 96% after two epochs while the validation loss rises: the model starts to memorise the training scans. The per-slice split hid this completely, because its validation set was made of near-copies of training slices. The model with the best validation accuracy (epoch 10) is kept.

### Where the model looks

[Grad-CAM](https://arxiv.org/abs/1610.02391) highlights the regions of an image that drove the prediction (red = most important).

![Grad-CAM heatmaps on test slices](assets/gradcam.png)

The first three slices are classified correctly. On the COVID-positive slice the model focuses on the lesion at the back of the left lung, and on the non-informative slice on the neck. The other five are COVID-positive slices predicted as normal. On `pCT1` and `pCT10` the opacities at the back of the lungs are clearly visible, but the model's attention is elsewhere. Both slices come from a scan with a circular field of view, which suggests the model is sensitive to how the images were acquired.

## How it works

- **Data:** the slices are grouped into pseudo-scans (see above) and whole groups are split into about 70% training, 15% validation and 15% test, stratified by class. Training images are augmented with random resized crops and horizontal flips. All images are normalized with the ImageNet statistics.
- **Model:** MobileNetV3-Large pretrained on ImageNet. Its original 1,000-class head is replaced by a new one: 960 → 1,280 (ReLU, 20% dropout) → 3. The whole network is fine-tuned (4.2M trainable parameters).
- **Training:** cross-entropy loss, AdamW (learning rate 1e-4, weight decay 1e-4), batch size 16, 25 epochs. The model with the best validation accuracy is kept.
- **Hardware:** Apple Silicon MacBook, PyTorch MPS backend. Training took about 85 minutes.

## Project structure

```
covid_ct/           shared code: settings, splits, data, model, training loops, metrics, Grad-CAM
find_groups.py      groups near-identical slices and writes the per-scan split
check_leakage.py    measures how much the test set leaks from the training set
train.py            trains the model and saves it with its history and curves
evaluate.py         per-class metrics, sensitivity, specificity and a per-scan summary
gradcam.py          Grad-CAM heatmaps of test images or of your own images
predict.py          classifies one or more images
splits/             the per-scan split (group_split.csv)
classifier.ipynb    the original pipeline step by step, with the per-slice split
```

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

Then run the scripts from the project folder:

```bash
python check_leakage.py                   # leakage of the per-slice and per-scan splits
python train.py                           # train on the per-scan split (about 1.5 hours on Apple Silicon)
python evaluate.py                        # evaluate the model on the test scans
python gradcam.py --examples 1 --errors   # heatmaps, saved to assets/gradcam.png
python predict.py data/pCT/pCT1227.jpg    # classify one or more images
```

Add `--split-mode slice` to `train.py` or `evaluate.py` to reproduce the original per-slice results. The per-scan split is included in `splits/group_split.csv`; run `find_groups.py` only to change the grouping threshold.

## Limitations

- **Groups are only a proxy for patients.** Without patient IDs, distant slices of the same patient that are not linked could still end up on both sides of the split. The test set contains only 29 scans, 8 of them with COVID-positive slices, so the results are uncertain.
- **One data source.** All scans come from two hospitals in Wuhan. The model has not been tested on other scanners or populations, and the Grad-CAM errors suggest it is sensitive to how images are acquired.
- **Overconfident probabilities.** The model often gives 99–100% even when it is wrong; its probabilities are not calibrated.
- **Single slices only.** A real diagnosis looks at the whole scan together with symptoms and lab results.

## Acknowledgements

- Dataset: [CT Scans for COVID-19 Classification](https://www.kaggle.com/datasets/azaemon/preprocessed-ct-scans-for-covid19) (CC BY 4.0), built from the iCTCF data of Ning et al. (2020).
- Model: Howard et al., "Searching for MobileNetV3", ICCV 2019. Pretrained weights from torchvision.
- Grad-CAM: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017.
