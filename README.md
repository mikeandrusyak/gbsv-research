# GBSV Research — Signal & Image Processing

Coursework for the FHNW module "Grundlagen der Bild- und Signalverarbeitung" (fundamentals of image and signal processing), organized as two mini-challenges with six fully documented experiments.

**Rendered reports:** https://mikeandrusyak.github.io/gbsv-research/

## Mini-Challenge 1 — Signal Processing (`mc1/`)

Analysis of a synthetic 1D train-bogie vibration signal on a fixed 1.0 s window:

1. `mc1/sampling_theorem.ipynb` — sampling theorem, aliasing, and reconstruction
2. `mc1/correlation.ipynb` — defect localization via cross-correlation
3. `mc1/convolution.ipynb` — convolution and deconvolution of sensor smearing

## Mini-Challenge 2 — Image Processing (`mc2/`)

Classical image-processing methods applied to my own photographs of Swiss subjects:

1. `mc2/augmentation.ipynb` — augmentation pipeline on a Geneva road-signs photo
2. `mc2/pattern_detection.ipynb` — template matching and denomination classification on Swiss coins
3. `mc2/segmentation.ipynb` — segmentation of Emmental cheese eyes with lighting correction

## Data Provenance Note

The synthetic signal generator is configured from assumptions inspired by the data description from:
https://www.kaggle.com/datasets/tamaryovell/predictive-maintanace-train-bogie-vibrations

The generated metadata file (`data/synthetic_defect_signal_meta.json`) includes a source description and source URL for traceability. All photographs in `data/` used for MC2 are my own.

## Environment Setup

1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Generate Synthetic Data (MC1)

To regenerate the synthetic signal and companion files:

```bash
python generate_synthetic_signal.py
```

Optional arguments:

```bash
python generate_synthetic_signal.py --seed 26 --noise-std 0.02 --out data/synthetic_defect_signal.npy
```

This command writes:
- signal array (`.npy`)
- event timestamps (`_event_times.npy`)
- metadata JSON (`_meta.json`)

## Run Notebooks

Open notebooks in `mc1/` and `mc2/` and run cells top-to-bottom.

MC1 notebooks assume:
- sampling rate: 575 Hz
- full duration: 60 s
- analysis window: first 1.0 s

## Reproducibility

- Default random seed is set to `26` in the generator.
- To reproduce previous results exactly, keep seed and parameters unchanged and rerun notebook cells from the beginning.

## Rendered HTML Reports (`docs/`)

The `docs/` folder contains the GitHub Pages site: a landing page (`docs/index.html`) and HTML exports of all six notebooks (`docs/mc1/`, `docs/mc2/`).
