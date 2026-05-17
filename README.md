# AO2D

AO2D is a PyTorch codebase for two-dimensional adaptive optics microscopy image restoration. It includes physics-based point spread function (PSF) generation, Zernike wavefront modeling, supervised restoration, and self-supervised restoration with an optical forward model.

The code is organized for 2-D image workflows while keeping model families commonly used in microscopy restoration and adaptive optics correction.

## Features

- 2-D optical forward model with Zernike wavefront aberrations.
- PSF generation from OSA/ANSI Zernike coefficients.
- Supervised training for `aberrated -> corrected` image restoration.
- Self-supervised training through `restored image + predicted Zernike coefficients -> re-aberrated image`.
- Multiple 2-D baseline models:
  - CARE2D
  - SCARE2D
  - RCAN2D
  - DFCAN2D
  - SFENet2D
  - PICNet2D

## Repository Structure

```text
src/ao2d/
  optics/        PSF, Zernike, wavefront, and FFT image formation
  data/          image I/O, paired datasets, and self-supervised datasets
  models/        CARE2D, SCARE2D, RCAN2D, DFCAN2D, SFENet2D, PICNet2D
  training/      forward model, losses, PSNR, and SSIM

scripts/
  train_supervised.py         supervised training
  train_self_supervised.py    self-supervised training
  test.py                     inference
  prepare_dataset.py          optional synthetic-data utility

configs/
  supervised_2d.json
  self_supervised_2d.json
  model_care2d.json
  model_scare2d.json
  model_rcan2d.json
  model_dfcan2d.json
  model_sfenet2d.json
  model_picnet2d.json
```

## Installation

```bash
python -m pip install -e .
```

The scripts also insert `src/` into `PYTHONPATH`, so they can be run directly from the repository root during development.

## Model Selection

All models can be created through one factory function:

```python
from ao2d.models.factory import make_model

model = make_model({"name": "rcan2d"})
```

Supported model names:

```text
care2d, scare2d, rcan2d, dfcan2d, sfenet2d, picnet2d
```

Model configuration examples are provided in `configs/model_*.json`.

## Optical Model

The optical model uses micrometer units throughout.

Main conventions:

- Zernike modes use OSA/ANSI single-index notation.
- Default Zernike indices are `3:15`, excluding piston, tip, and tilt.
- Zernike coefficients represent optical path difference amplitudes in micrometers.
- Wavefront phase is computed as:

```text
phase = 2*pi/lambda * wavefront
```

- The scalar 2-D PSF is generated from a circular pupil in frequency space:

```text
field = fftshift(ifft2(ifftshift(pupil)))
psf = abs(field)^2
```

- Image formation uses FFT convolution with `ifftshift(psf)`.

## Dataset Format

The recommended workflow is to prepare data externally and point the training configs to the generated images and manifest.

Expected directory layout:

```text
DATA_ROOT/
  OBJ/
  No_abe/
  abe/
  PSF/No_abe/
  PSF/abe/
  Zernike/
  metadata/manifest.csv
```

For supervised training, `metadata/manifest.csv` must contain at least:

```text
abe_path,no_abe_path
```

Optional columns such as `zernike_path`, `psf_abe_path`, and `rms_waves` can be included for bookkeeping.

If no manifest is available, paired directory loading can be used with:

```text
abe/      input aberrated images
No_abe/   target corrected images
```

## Supervised Training

Edit `configs/supervised_2d.json` so the manifest path points to your dataset:

```json
"data": {
  "patch_size": [256, 256],
  "train": {
    "manifest": "/path/to/DATA_ROOT/metadata/manifest.csv",
    "augment": true,
    "samples_per_epoch": 1000
  }
}
```

Run:

```bash
python scripts/train_supervised.py \
  -c configs/supervised_2d.json \
  -o outputs/care2d_supervised
```

The default supervised model is `care2d`. To use another baseline, replace the `model` section in the config with one of the `configs/model_*.json` examples.

## Self-Supervised Training

Self-supervised training only requires aberrated input images. Models used in this mode must return both a restored image and Zernike coefficients; `scare2d` and `picnet2d` satisfy this interface.

Edit `configs/self_supervised_2d.json`:

```json
"data": {
  "patch_size": [256, 256],
  "train": {
    "image_dir": "/path/to/DATA_ROOT/abe",
    "augment": true,
    "samples_per_epoch": 1000
  }
}
```

Run:

```bash
python scripts/train_self_supervised.py \
  -c configs/self_supervised_2d.json \
  -o outputs/scare2d_self_supervised
```

Training objective:

```text
aberrated image
  -> model -> restored image + Zernike coefficients
  -> AO forward model -> estimated aberrated image
  -> loss(estimated aberrated, input aberrated)
```

## Inference

```bash
python scripts/test.py \
  --checkpoint outputs/care2d_supervised/best.pt \
  --input data/ao2d/abe \
  --output outputs/test_results
```

If the model predicts Zernike coefficients, `test.py` also saves `*_zernike_coeff_um.txt`.

## PICNet2D vs SCARE2D

PICNet2D and SCARE2D both support physics-guided adaptive optics restoration, but their modeling assumptions are different.

| Aspect | PICNet2D | SCARE2D |
| --- | --- | --- |
| Network structure | Separate object generator and aberration regressor | One restoration backbone with a Zernike regression branch |
| Output | Restored image and predicted Zernike coefficients | Restored image and predicted Zernike coefficients |
| Training style | Suits staged or hybrid training with synthetic supervision and physical consistency | Suits direct end-to-end self-supervised training |
| Complexity | More modular and easier to extend with extra constraints | Simpler and easier to train as a single model |
| Typical use | When synthetic object/aberration labels are available or desired | When only aberrated images and a forward model are available |

In short, PICNet2D is a more modular physical inverse-modeling framework, while SCARE2D is a compact end-to-end self-supervised restoration model.

## Notes

- Keep all optical distances and wavelengths in micrometers.
- Use image patches large enough for the chosen model depth.
- Self-supervised training quality depends strongly on the accuracy of the forward optical model.

