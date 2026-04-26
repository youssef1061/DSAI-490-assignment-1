# DSAI-490 — Representation Learning with Autoencoders

**Student:** Youssef Hatem — 202201596

Autoencoder (AE) and Variational Autoencoder (VAE) trained on the
**Medical-MNIST** dataset (6 anatomical regions: AbdomenCT, BreastMRI,
CXR, ChestCT, Hand, HeadCT).

---

## Project Structure

```
dsai490/                          ← upload this whole folder to MyDrive
├── data/
│   ├── raw/
│   │   └── medical-mnist.zip     ← upload your dataset zip here
│   └── processed/                ← (auto-created, currently unused)
├── models/                       ← best checkpoints saved here automatically
├── notebooks/
│   └── dsai490_colab.ipynb       ← main experiment notebook (run in Colab)
├── outputs/                      ← figures saved here automatically
├── src/
│   ├── __init__.py
│   ├── data_processing.py        ← tf.data pipeline
│   ├── model.py                  ← AE and VAE definitions
│   ├── train.py                  ← training loop utilities
│   └── visualize.py              ← all plotting functions
├── tests/
│   ├── test_data_processing.py
│   └── test_model.py
├── README.md
└── requirements.txt
```

---

## Setup — Google Colab

### 1. Upload files to Google Drive

Upload the entire `dsai490/` folder to `My Drive` so the path becomes:

```
My Drive/dsai490/
```

Make sure `data/raw/medical-mnist.zip` is present before running the notebook.

### 2. Open the notebook

Open `notebooks/dsai490_colab.ipynb` in Google Colab.

Enable GPU: **Runtime → Change runtime type → T4 GPU**.

### 3. Run all cells

Cell 0 mounts Google Drive and extracts the dataset automatically.
All subsequent cells import from `src/` and write outputs to
`outputs/<region>/`.

---

## Running Tests (optional, local only)

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## Code Conventions

This project follows the course Code Conventions:

- **PEP 8** — enforced via `flake8` / `black`
- **PEP 257** — all modules, classes, and functions have docstrings
- **Type hints** — every function signature is fully annotated
- **Imports** — standard library → third-party → local
- **tf.data** — all data loading uses `tf.data.Dataset` pipelines
- **Model versioning** — checkpoints include a version tag (e.g. `_v1`)
- **Modular code** — logic split across `data_processing`, `model`,
  `train`, and `visualize` modules; the notebook only calls these

---

## Results

Outputs (loss curves, reconstructions, latent-space plots, generated
samples, denoising comparisons) are saved per region under `outputs/`.

| Metric | AE | VAE |
|---|---|---|
| Reconstruction sharpness | Higher | Slightly blurry |
| Latent space structure | Fragmented | Smooth / continuous |
| Generation from prior | ✗ | ✓ |
| Denoising | Good | Good (smoother) |
