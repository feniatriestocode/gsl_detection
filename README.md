# GSL Detection

**Isolated Greek Sign Language (GSL) recognition from video, using MediaPipe landmarks and a Transformer encoder.**

[ ![CCDS](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-landmarks-0097A7)

A video of a single signed gloss goes in and a predicted gloss comes out. Instead of feeding raw pixels to a network, the pipeline extracts body and hand landmarks with MediaPipe, normalizes them so the signer's position and size don't matter, and classifies the resulting sequence with a small Transformer encoder.

## Results

Evaluated on a **signer-independent** test set: the three test signers never appear in training or validation.

| Metric | Score |
|---|---|
| Top-1 accuracy | **68.34%** |
| Top-5 accuracy | **93.55%** |
| Weighted F1 | **0.669** |
| Classes | 369 |
| Test samples | 3,318 |

Per-class numbers are in [`reports/per_class_metrics.csv`](reports/per_class_metrics.csv), the raw metrics in [`reports/test_metrics.json`](reports/test_metrics.json), and plots (training curves, confusion pairs, per-class accuracy) in [`reports/figures/`](reports/figures). The full write-up is [`reports/Project_Report.pdf`](reports/Project_Report.pdf).

## How it works

```
 video (.mp4)
     │
     ▼
 1. Landmark extraction      MediaPipe Pose (lite) + Hand Landmarker  →  186 values / frame
     │
     ▼
 2. Normalization            shoulder-relative centering & scaling  →  z-score (train stats)  →  pad/truncate to 90 frames
     │
     ▼
 3. Transformer encoder      linear projection → positional encoding → Pre-LN encoder layers → masked mean pooling
     │
     ▼
 4. Classifier head          Linear → GELU → Dropout → Linear  →  gloss
```

### 1. Features (186 per frame)

| Block | Landmarks | Values |
|---|---|---|
| Upper-body pose | 15 | 15 × 4 (x, y, z, visibility) = 60 |
| Left hand | 21 | 21 × 3 = 63 |
| Right hand | 21 | 21 × 3 = 63 |

Face landmarks are deliberately excluded. Missing detections are zero-filled, and hands are assigned to the left or right slot by MediaPipe's handedness label.

### 2. Normalization

1. **Pose-relative:** every landmark is centered on the midpoint of the shoulders and divided by the shoulder distance, removing the effect of where the signer stands and how far they are from the camera.
2. **Z-score:** per-feature mean and standard deviation are computed on the *training* signers only (Welford's online algorithm) and applied to all splits.
3. **Pad / truncate** to 90 frames, with a boolean mask so padding is ignored by attention and pooling.

### 3. Model

`GSLTransformer` is a Pre-LN Transformer encoder with sinusoidal positional encoding and masked mean pooling. Default configuration used for training and evaluation:

| Setting | Value |
|---|---|
| `d_model` | 256 |
| Attention heads | 4 |
| Encoder layers | 3 |
| Feed-forward dim | 1024 (4 × `d_model`) |
| Dropout | 0.5 |
| Loss | Cross-entropy, label smoothing 0.1 |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-4 |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 10) |
| Batch size | 64 |
| Early stopping | patience 20 on validation loss |

### 4. Augmentation

Applied online to training sequences only (each with probability 0.5): Gaussian noise, temporal warping (0.8–1.2×), horizontal flip with left/right hand swap, small in-plane rotation, and random scaling. See [`gsl_detect/augment.py`](gsl_detect/augment.py).

## Dataset

The project uses the isolated part of the GSL SL-ReDu corpus (https://sl-redu.e-ce.uth.gr/el/corpus/) Developed @ the University of Thessaly (21 signers, one gloss per video) with a CSV listing each video and its annotation. Videos are expected at:

```
data/raw/isolated_GSL_corpus/
├── isolated_GSL_corpus.csv
└── Glosses_videos/
    ├── signer01/….mp4
    ├── signer02/….mp4
    └── …
```

The `data/` folder is git-ignored, so you need to obtain the corpus yourself and respect its license.

**Splits** are made by signer, with a fixed seed (42): 3 test signers, 3 validation signers, and the remaining signers for training. The label encoder is fit on the training set only.

## Getting started

Requires Python 3.12.

```bash
git clone https://github.com/feniatriestocode/gsl_detection.git
cd gsl_detection

python -m venv .venv && source .venv/bin/activate
pip install -e .    # dependencies are declared in pyproject.toml
pip install torchmetrics tensorboard   # imported by training/evaluation, not yet declared in pyproject.toml
```

> **Run everything from the repository root.** The MediaPipe `.task` model files are loaded by relative path.

### Run the full pipeline

```bash
# 1. Extract landmarks from every video -> data/interim/<signer>/<video>_proc.npy
python -m gsl_detect.features

# 2. Normalize and split -> data/processed/{train,val,test}/…_norm.npz
python -m gsl_detect.normalization

# 3. Make the label CSV available where training expects it
cp data/raw/isolated_GSL_corpus/isolated_GSL_corpus.csv data/processed/

# 4. (optional) Sanity-check the processed data
python -m gsl_detect.sanity_check

# 5. Train -> models/best_model.pt, TensorBoard logs in runs/
python -m gsl_detect.modeling.train

# 6. Evaluate on the held-out signers -> reports/ and reports/figures/
python -m gsl_detect.modeling.evaluate
```

Training and evaluation are [Typer](https://typer.tiangolo.com/) CLIs, so hyperparameters can be overridden, e.g.:

```bash
python -m gsl_detect.modeling.train --d-model 256 --num-layers 3 --batch-size 64 --lr 1e-4
python -m gsl_detect.modeling.evaluate --help
```

Monitor training with `tensorboard --logdir runs`.

The evaluation plots use LaTeX for text rendering, so a working LaTeX installation is needed to generate the figures.

### Demo app

```bash
streamlit run app.py
```

- **Upload Video Clip:** upload a short clip of a single sign and get the predicted gloss.
- **Live Continuous Stream:** webcam prediction. *Experimental and not working reliably yet.*

The app rebuilds the model from `models/best_model.pt`, so the architecture settings in `app.py` must match the checkpoint you trained.

## Repository structure

```
├── app.py                  Streamlit demo (video upload + experimental live webcam)
├── gsl_detect/
│   ├── config.py           Paths, feature layout, constants
│   ├── features.py         MediaPipe landmark extraction (multiprocessing by signer group)
│   ├── normalization.py    Pose-relative normalization, z-score, padding, signer-based splits
│   ├── augment.py          Online augmentation
│   ├── sanity_check.py     Dataset statistics and landmark animation
│   ├── plots.py            Plotting helpers
│   └── modeling/
│       ├── dataset.py      Dataset, collate function, dataloaders
│       ├── model.py        GSLTransformer
│       ├── train.py        Training loop with early stopping
│       ├── evaluate.py     Test metrics, confusion analysis, figures
│       └── predict.py      Inference entry point (placeholder)
├── models/                 Trained checkpoint (best_model.pt)
├── notebooks/              Exploration notebooks
├── reports/                Metrics, figures, project report
├── runs/                   TensorBoard logs
├── *.task                  MediaPipe model bundles (pose, hand, face, holistic)
├── Makefile                Helper commands (`make help`)
└── pyproject.toml          Package metadata and ruff configuration
```

## Known limitations

- **Isolated signs only.** The model classifies one gloss per clip; it does not segment or translate continuous signing.
- **Domain gap.** Training data comes from a controlled recording setup, so performance on casual smartphone video is expected to be lower.
- **No facial or mouth features.** Non-manual markers, which carry meaning in sign languages, are not used.
- **Live webcam tab is unfinished.** The streaming code path in `app.py` is still being debugged.
- **`modeling/predict.py` is a stub.** Inference currently goes through `app.py` or `evaluate.py`.


## Author

Fenia Papadouli, Department of Electrical and Computer Engineering, University of Thessaly.
