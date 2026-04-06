# GBSV Research - MC1 Signal Processing

This repository contains the MC1 coursework for synthetic train bogie vibration analysis:
- Sampling Theorem
- Correlation
- Convolution and Deconvolution

The project uses a synthetic 1D vibration signal and evaluates signal-processing methods on a fixed 1.0 s analysis window.


## Data Provenance Note

The synthetic signal generator is configured from assumptions inspired by the data description from:
https://www.kaggle.com/datasets/tamaryovell/predictive-maintanace-train-bogie-vibrations

The generated metadata file (`data/synthetic_defect_signal_meta.json`) includes a source description and source URL for traceability.

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

## Generate Synthetic Data

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

Open notebooks in `mc1/` and run cells top-to-bottom:

1. `mc1/sampling_theorem.ipynb`
2. `mc1/correlation.ipynb`
3. `mc1/convolution.ipynb`

All notebooks assume:
- sampling rate: 575 Hz
- full duration: 60 s
- analysis window: first 1.0 s

## Reproducibility

- Default random seed is set to `26` in the generator.
- To reproduce previous results exactly, keep seed and parameters unchanged and rerun notebook cells from the beginning.
