# Binary AI-Art Detection

Reproducible PyTorch coursework project for classifying human-created art
(`0`) versus AI-generated art (`1`) with
[AI-ArtBench](https://www.kaggle.com/datasets/ravidussilva/real-ai-art).
The project preserves the dataset's official test partition and uses a fixed,
balanced 10,000-image protocol.

The implementation includes:

- MobileNetV2 baselines with and without augmentation;
- frozen and partially fine-tuned ConvNeXt-Tiny;
- a final-feature squeeze-and-excitation ablation;
- accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC curves;
- style metrics and class-appropriate source error rates;
- contrast, JPEG, and common-resampling robustness tests;
- Grad-CAM panels, a reproducibility manifest, an executed notebook workflow,
  and a LaTeX report.

The SE model is a simplified paper-inspired ablation. It does not reproduce or
claim the paper's multi-level AttentionConvNeXt architecture.

## 1. Environment

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,data]"
```

Install the CUDA-specific PyTorch build first if the default wheel is not
appropriate for the machine.

## 2. Download and validate AI-ArtBench

The downloader pins Kaggle dataset version 5 and extracts it below
`data/raw/real-ai-art/`. The archive is about 10.66 GB; at least 25 GiB of free
space is required by the safety check.

Public-dataset access may work anonymously. If Kaggle requests authentication,
create a token in Kaggle's API settings and expose it only for the current
shell:

```bash
export KAGGLE_API_TOKEN="your-token"
python scripts/download_dataset.py
unset KAGGLE_API_TOKEN
```

Do not place tokens in this repository. Images, credentials, checkpoints, and
generated outputs are excluded by `.gitignore`.
Review the dataset card's specified license before use and do not redistribute
the downloaded images with the coursework source.

After extraction, the script requires exactly 185,015 readable image paths:

| Official split | Human | Latent Diffusion | Stable Diffusion |
|---|---:|---:|---:|
| Train | 50,000 | 52,092 | 52,923 |
| Test | 10,000 | 10,000 | 10,000 |

Unexpected folder aliases or counts cause a failure instead of silently
changing the experiment.

## 3. Fixed coursework protocol

Seed 42 selects exact source/style quotas:

| Split | Human | Latent Diffusion | Stable Diffusion | Total |
|---|---:|---:|---:|---:|
| Train | 3,200 | 1,600 | 1,600 | 6,400 |
| Validation | 800 | 400 | 400 | 1,600 |
| Test | 1,000 | 500 | 500 | 2,000 |

Each number is distributed equally across the ten styles. Training and
validation come only from the official training pool; test images come only
from the official test pool. Paths are persisted and checked for overlap.

Run a short pipeline check:

```bash
python scripts/run_experiments.py \
  --data-root data/raw/real-ai-art \
  --epochs 1 \
  --experiments E0_mobilenetv2_noaug_frozen
```

Run the final E0--E4 suite:

```bash
python scripts/run_experiments.py \
  --data-root data/raw/real-ai-art
```

Defaults are 224×224 images, batch size 32, eight epochs, AdamW, seed 42,
validation-F1 early stopping, and a fixed 0.5 threshold. The best model is
chosen by validation F1, never by test performance.
If an interrupted run already contains complete per-experiment artifacts,
repeat the command with `--resume`.

## 4. Independent replication audit

Evaluate all five saved checkpoints on a second, disjoint 2,000-image holdout:

```bash
python scripts/evaluate_replication.py
```

The replication split uses seed 4242 and the same per-source/per-style quotas
as the primary test set. It excludes every original train, validation, and test
path. This audit uses only the clean direct-resize 224×224 evaluation transform;
it never invokes or combines the 128-to-224 robustness resampling condition.
Outputs are isolated beneath `outputs/replication/`.

The resulting table compares clean-training, primary-test, and replication F1
for E0-E4, with source/style-stratified bootstrap confidence intervals. It is a
same-dataset overfitting and stability check, not an unseen-generator test.

## 5. Notebook and report

The notebook mirrors the complete workflow:

```bash
jupyter lab notebooks/01_ai_art_detection.ipynb
```

After the five experiments finish, generate the measured LaTeX fragments and
compile the report:

```bash
python scripts/make_latex_table.py
cd report
tectonic report.tex
```

`report/report.tex` deliberately refuses to compile without measured result
fragments. Replace `Your Name` before submission.

## 6. Verification

```bash
pytest
ruff check .
```

Tests do not require the Kaggle dataset or pretrained downloads. They cover
metadata inference, strict quota sampling, split isolation, metrics,
robustness transforms, model output shape, and a synthetic training smoke
test.

## Experiment matrix

| ID | Model | Trainable part | Augmentation | Question |
|---|---|---|---|---|
| E0 | MobileNetV2 | classifier | no | fast baseline |
| E1 | MobileNetV2 | classifier | yes | augmentation effect |
| E2 | ConvNeXt-Tiny | classifier | yes | backbone comparison |
| E3 | ConvNeXt-Tiny | last stage + classifier | yes | fine-tuning effect |
| E4 | ConvNeXt-Tiny + SE | last stage + SE + classifier | yes | attention ablation |

## Repository map

```text
src/ai_art_detection/   data, model, training, evaluation, and Grad-CAM code
notebooks/              end-to-end coursework notebook
scripts/                download, experiment, and report-generation commands
tests/                  dataset-independent verification
report/                 LaTeX source and bibliography
outputs/                generated metrics, figures, checkpoints, and splits
```
