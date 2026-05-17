# AO2D

AO2D is a PyTorch codebase for two-dimensional adaptive optics microscopy image restoration. It includes physics-based point spread function (PSF) generation, Zernike wavefront modeling, supervised restoration, and self-supervised restoration with an optical forward model.

The code is organized for 2-D image workflows while keeping model families commonly used in microscopy restoration and adaptive optics correction.

## Features

- 2-D optical forward model with Zernike wavefront aberrations.
- PSF generation from OSA/ANSI Zernike coefficients.
- Supervised training for `aberrated -> corrected` image restoration.
- Self-supervised training through an optical reconstruction cycle.
- Proposed model:
  - SCARE2D
- Baseline models:
  - CARE2D
  - RCAN2D
  - DFCAN2D
  - SFENet2D
  - PICNet2D

## Repository Structure

```text
src/ao2d/
  optics/        PSF, Zernike, wavefront, and FFT image formation
  data/          image I/O, paired datasets, and self-supervised datasets
  models/        SCARE2D and 2-D baseline models
  training/      forward model, losses, PSNR, and SSIM

scripts/
  train_supervised.py         supervised training
  train_self_supervised.py    self-supervised training
  train_two_stage.py          two-stage PICNet2D-style training
  test.py                     inference
  prepare_dataset.py          optional synthetic-data utility

configs/
  supervised_2d.json
  self_supervised_2d.json
  two_stage_2d.json
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

## Models

All models can be created through one factory function:

```python
from ao2d.models.factory import make_model

model = make_model({"name": "scare2d"})
```

The proposed model is:

```text
scare2d
```

Baseline model names:

```text
care2d, rcan2d, dfcan2d, sfenet2d, picnet2d
```

Model configuration examples are provided in `configs/model_*.json`.

## Optical Model

The optical model uses micrometer units throughout.

Main conventions:

- Zernike modes use OSA/ANSI single-index notation.
- Default Zernike indices are `3:15`, excluding piston, tip, and tilt.
- Zernike coefficients represent optical path difference amplitudes in micrometers.
- The wavefront is represented as a weighted Zernike expansion:

$$
W(\rho,\theta)=\sum_{j \in \mathcal{J}} c_j Z_j(\rho,\theta),
$$

where \(c_j\) is the optical path difference coefficient in micrometers and \(\mathcal{J}=\{3,\ldots,15\}\) by default.

- Wavefront phase is computed as:

$$
\phi(\rho,\theta)=\frac{2\pi}{\lambda}W(\rho,\theta).
$$

- The scalar 2-D PSF is generated from a circular pupil in frequency space:

$$
P(\rho,\theta)=A(\rho)\exp\left(i\phi(\rho,\theta)\right),
$$

$$
E=\mathcal{F}^{-1}\{P\},
\qquad
h=\frac{|E|^2}{\sum |E|^2}.
$$

- Image formation uses FFT convolution with `ifftshift(psf)`.

For an object image \(x\) and PSF \(h\), the aberrated image is modeled as:

$$
y = x * h.
$$

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
  validation/
    OBJ/
    No_abe/
    abe/
    PSF/No_abe/
    PSF/abe/
    Zernike/
    metadata/manifest.csv
```

The top-level folders are used for training. `DATA_ROOT/validation/` is used as a dedicated validation set and should follow the same internal layout.

For supervised training, `metadata/manifest.csv` must contain:

| Column | Required | Description |
| --- | --- | --- |
| `abe_path` | yes | Path to the aberrated input image. |
| `no_abe_path` | yes | Path to the corrected or no-aberration target image. |
| `object_path` | no | Path to the underlying object image, if available. |
| `zernike_path` | no | Path to a coefficient file, if available. |
| `psf_abe_path` | no | Path to the aberrated PSF file. |
| `psf_no_abe_path` | no | Path to the no-aberration PSF file. |
| `rms_waves` | no | Aberration RMS level in waves. |
| `rms_nm` | no | Aberration RMS level in nanometers. |
| `mode` | no | Microscopy mode, e.g. `widefield`. |
| `pixel_size_um` | no | Pixel size in micrometers. |
| `na` | no | Numerical aperture. |
| `lambda_em_um` | no | Emission wavelength in micrometers. |
| `lambda_ex_um` | no | Excitation wavelength in micrometers. |

Minimal example:

```text
abe_path,no_abe_path
data/ao2d/abe/sample_001.tif,data/ao2d/No_abe/sample_001.tif
```

Paths may be absolute or relative to the manifest directory.

If no manifest is available, paired directory loading can be used with:

```text
abe/      input aberrated images
No_abe/   target corrected images
```

## Supervised Training

Point the code to your dataset root with either `--data-root`, `AO2D_DATA_ROOT`, or
`data.root`/`data.data_root` in the config. Keep paths inside the config relative to
that root so the same config works on different machines:

```json
"data": {
  "patch_size": [256, 256],
  "train": {
    "manifest": "metadata/manifest.csv",
    "augment": true,
    "samples_per_epoch": 1000
  },
  "val": {
    "manifest": "validation/metadata/manifest.csv",
    "augment": false,
    "samples_per_epoch": 100
  }
}
```

Run:

```bash
python scripts/train_supervised.py \
  -c configs/supervised_2d.json \
  --data-root /path/to/DATA_ROOT \
  -o outputs/care2d_supervised
```

Or set the environment variable once on each server:

```bash
export AO2D_DATA_ROOT=/path/to/DATA_ROOT
python scripts/train_supervised.py -c configs/supervised_2d.json -o outputs/care2d_supervised
```

Multi-GPU training is supported with `torchrun`:

```bash
torchrun --nproc_per_node=2 scripts/train_supervised.py \
  -c configs/supervised_2d.json \
  --data-root /mnt/share/dyx/Data/Data2d \
  -o outputs/care2d
```

The default supervised configuration uses `care2d` as a baseline. To use another baseline, replace the `model` section in the config with one of the `configs/model_*.json` examples.

Optimizer and learning-rate decay are configured in the `training` block. The field names are compatible with the original SSPAO configs. The supervised baseline uses Adam with StepLR by default:

```json
"training": {
  "optimizer": "Adam",
  "initial_learning_rate": 0.0001,
  "lr_scheduler": "StepLR",
  "step_size": 25,
  "gamma": 0.5
}
```

Supported optimizers are `Adam`, `AdamW`, and `SGD`. Supported schedulers are `StepLR`, `CosineAnnealingLR`, `MultiStepLR`, `ReduceLROnPlateau`, and `none`. The shorter names `step`, `cosine`, `multistep`, and `plateau` are also accepted.

## Self-Supervised Training

Self-supervised training only requires aberrated input images. The default self-supervised model is SCARE2D, which predicts both a restored image and Zernike coefficients. PICNet2D also satisfies this output interface and can be used as a baseline.

Edit `configs/self_supervised_2d.json`:

```json
"data": {
  "patch_size": [256, 256],
  "train": {
    "image_dir": "/path/to/DATA_ROOT/abe",
    "augment": true,
    "samples_per_epoch": 1000
  },
  "val": {
    "image_dir": "/path/to/DATA_ROOT/validation/abe",
    "samples_per_epoch": 100
  }
}
```

Run:

```bash
python scripts/train_self_supervised.py \
  -c configs/self_supervised_2d.json \
  -o outputs/scare2d_self_supervised
```

Multi-GPU training:

```bash
torchrun --nproc_per_node=4 scripts/train_self_supervised.py \
  -c configs/self_supervised_2d.json \
  -o outputs/scare2d_self_supervised
```

For an aberrated input \(y\), SCARE2D predicts a restored image \(\hat{x}\) and Zernike coefficients \(\hat{\mathbf{c}}\):

$$
(\hat{x},\hat{\mathbf{c}})=f_\theta(y).
$$

The predicted coefficients generate a PSF \(h(\hat{\mathbf{c}})\), which re-aberrates the restored image:

$$
\hat{y}=\hat{x} * h(\hat{\mathbf{c}}).
$$

The self-supervised reconstruction loss is:

$$
\mathcal{L}_{\mathrm{self}}
=
\left\|\hat{y}-y\right\|_1
+ \lambda_{\mathrm{TV}}\mathrm{TV}(\hat{x})
+ \lambda_c\left\|\hat{\mathbf{c}}\right\|_2^2.
$$

The default SCARE2D self-supervised config follows the SSPAO SCARE training logic:

```json
"training": {
  "optimizer": "Adam",
  "initial_learning_rate": 0.0001,
  "lr_scheduler": "ReduceLROnPlateau",
  "factor": 0.5,
  "patience": 15,
  "min_lr": 0.000004
}
```

## Two-Stage Training

`train_two_stage.py` implements a PICNet2D-style baseline. It trains two networks:

$$
\hat{x}=G_{\mathrm{obj}}(y),
\qquad
\hat{\mathbf{c}}=G_{\mathrm{aber}}(y),
$$

where \(G_{\mathrm{obj}}\) restores the object image and \(G_{\mathrm{aber}}\) estimates Zernike coefficients.

Compared with `train_self_supervised.py`, the two-stage trainer uses clean/object images to create synthetic aberrated examples with known coefficients:

$$
y_{\mathrm{synth}} = x * h(\mathbf{c}).
$$

Stage 1 performs synthetic supervised pretraining:

$$
\mathcal{L}_{\mathrm{stage1}}
=
\left\|\hat{y}_{\mathrm{synth}}-y_{\mathrm{synth}}\right\|_1
+ \alpha\left\|\hat{x}-x\right\|_1
+ \beta\left\|\hat{\mathbf{c}}-\mathbf{c}\right\|_2^2.
$$

Stage 2 mixes real-image physical consistency and synthetic supervision:

$$
\mathcal{L}_{\mathrm{stage2}}
=
\left\|\hat{y}_{\mathrm{real}}-y_{\mathrm{real}}\right\|_1
+ \gamma\mathcal{L}_{\mathrm{synth}}
+ \lambda_{\mathrm{TV}}\mathrm{TV}(\hat{x})
+ \lambda_c\left\|\hat{\mathbf{c}}\right\|_2^2.
$$

Two-stage training has separate learning-rate fields for the two phases:

```json
"training": {
  "optimizer": "Adam",
  "initial_learning_rate": 0.0001,
  "stage2_lr_scale": 0.1,
  "lr_scheduler_stage1": "StepLR",
  "lr_scheduler_stage2": "StepLR",
  "step_size_stage1": 12,
  "step_size_stage2": 25,
  "gamma": 0.5
}
```

Use this route when clean object images are available and you want a stronger synthetic-supervision baseline.

Edit `configs/two_stage_2d.json`:

```json
"data": {
  "patch_size": [256, 256],
  "train": {
    "object_dir": "/path/to/DATA_ROOT/OBJ",
    "aberrated_dir": "/path/to/DATA_ROOT/abe",
    "augment": true
  }
}
```

Run both stages:

```bash
python scripts/train_two_stage.py \
  -c configs/two_stage_2d.json \
  -o outputs/picnet2d_two_stage \
  --stage both
```

Multi-GPU two-stage training:

```bash
torchrun --nproc_per_node=4 scripts/train_two_stage.py \
  -c configs/two_stage_2d.json \
  -o outputs/picnet2d_two_stage \
  --stage both
```

Run only stage 1:

```bash
python scripts/train_two_stage.py \
  -c configs/two_stage_2d.json \
  -o outputs/picnet2d_two_stage \
  --stage stage1
```

Run stage 2 from a stage-1 checkpoint:

```bash
python scripts/train_two_stage.py \
  -c configs/two_stage_2d.json \
  -o outputs/picnet2d_two_stage \
  --stage stage2 \
  --resume_stage1 outputs/picnet2d_two_stage/stage1_best.pt
```

## Inference

```bash
python scripts/test.py \
  --checkpoint outputs/care2d_supervised/best.pt \
  --input data/ao2d/abe \
  --output outputs/test_results
```

If the model predicts Zernike coefficients, `test.py` also saves `*_zernike_coeff_um.txt`.

Two-stage checkpoints are also supported:

```bash
python scripts/test.py \
  --checkpoint outputs/picnet2d_two_stage/stage2_best.pt \
  --input data/ao2d/abe \
  --output outputs/picnet2d_two_stage/test_results
```

## PICNet2D Baseline vs SCARE2D

SCARE2D is the proposed model in this codebase. PICNet2D is included as a baseline with a more modular inverse-modeling design.

| Aspect | PICNet2D | SCARE2D |
| --- | --- | --- |
| Role in this repository | Baseline | Proposed model |
| Network structure | Separate object generator and aberration regressor | One restoration backbone with a Zernike regression branch |
| Output | Restored image and predicted Zernike coefficients | Restored image and predicted Zernike coefficients |
| Training script | `scripts/train_two_stage.py` | `scripts/train_self_supervised.py` |
| Training style | Staged or hybrid training with synthetic supervision and physical consistency | Direct end-to-end self-supervised training |
| Complexity | More modular and easier to extend with extra constraints | Simpler and easier to train as a single model |
| Typical use | When synthetic object/aberration labels are available or desired | When only aberrated images and a forward model are available |

In short, PICNet2D is a more modular physical inverse-modeling framework, while SCARE2D is a compact end-to-end self-supervised restoration model.

## Notes

- Keep all optical distances and wavelengths in micrometers.
- Use image patches large enough for the chosen model depth.
- `batch_size` in config files is the per-process batch size. With `torchrun --nproc_per_node=4`, the effective global batch size is `4 * batch_size`.
- In distributed training, only rank 0 writes checkpoints and logs.
- Self-supervised training quality depends strongly on the accuracy of the forward optical model.
