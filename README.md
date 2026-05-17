# AO2D

这是一个参考私有仓库 `YuxiDn/SSPAO` 后整理出的二维图像 AO 显微恢复代码。原仓库以 3D 为主，这里把对应思想改成 2D，并把目录重新整理成更直接的结构。

## 目录

```text
src/ao2d/
  optics/        PSF、Zernike、wavefront、FFT 成像
  data/          图像读取、成对数据集、自监督数据集
  models/        CARE2D、SCARE2D、RCAN2D、DFCAN2D、SFENet2D、PICNet2D
  training/      forward model、loss、PSNR、SSIM

scripts/
  prepare_dataset.py          可选备用：生成/整理 2D AO 数据
  train_supervised.py         监督训练：aberrated -> no_aberration
  train_self_supervised.py    自监督训练：restored + zernike -> aberrated
  test.py                     推理测试

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

## 模型对应关系

| SSPAO 原模型 | 2D 文件 | 说明 |
| --- | --- | --- |
| `model/care.py` | `src/ao2d/models/care2d.py` | CARE 风格 2D encoder-decoder |
| `model/SCARE.py` | `src/ao2d/models/scare2d.py` | CARE2D + Zernike 回归分支 |
| `model/rcan3d.py` | `src/ao2d/models/rcan2d.py` | Residual Channel Attention Network |
| `model/DFCAN.py` | `src/ao2d/models/dfcan2d.py` | Fourier Channel Attention baseline |
| `model/SFENet.py` | `src/ao2d/models/sfenet2d.py` | spatial branch + frequency branch + decoder |
| `model/PICNet.py` | `src/ao2d/models/picnet2d.py` | object generator、aberration regressor、discriminator |

统一入口：

```python
from ao2d.models.factory import make_model

model = make_model({"name": "rcan2d"})
```

`name` 可选：

```text
care2d, scare2d, rcan2d, dfcan2d, sfenet2d, picnet2d
```

## 光学计算

光学部分与本地 MATLAB 函数保持一致：

```text
/Users/yxdeng/Library/CloudStorage/OneDrive-个人/Doctor/0-Microscopy/Adaptive_Optics/Matlab/Stim/functions
```

一致点：

- Zernike 使用 OSA/ANSI 单索引。
- 默认模式为 `3:15`，排除 piston、tip、tilt。
- Zernike 系数单位是微米 OPD。
- phase = `2*pi/lambda * wavefront`。
- PSF 使用频域圆形 pupil：
  `fftshift(ifft2(ifftshift(pupil)))`。
- FFT 成像卷积使用 `fft2(ifftshift(psf))`。

## 安装

```bash
python -m pip install -e .
```

没有安装包也可以直接运行 `scripts/`，脚本会自动把 `src` 加到 Python path。

## 使用 MATLAB 生成的数据

主流程假设数据由 MATLAB 生成，不需要 Python 再生成数据。监督训练最推荐使用 MATLAB 导出的：

```text
MATLAB_DATA_ROOT/
  OBJ/
  No_abe/
  abe/
  PSF/No_abe/
  PSF/abe/
  Zernike/
  metadata/manifest.csv
```

`metadata/manifest.csv` 至少需要包含：

```text
abe_path,no_abe_path
```

如果使用你当前 MATLAB `generateAO2DDatasetBatch.m` 生成的数据，它已经包含这些列，也包含 `zernike_path`、`psf_abe_path`、`rms_waves` 等额外信息，可以直接被 `AO2DPairDataset.from_manifest(...)` 读取。

如果没有 manifest，也可以直接使用目录：

```text
abe/      输入：带像差图像
No_abe/   标签：无像差/校正目标图像
```

## 监督训练

修改 [configs/supervised_2d.json](configs/supervised_2d.json) 里的 manifest 路径，让它指向 MATLAB 数据：

```json
"data": {
  "patch_size": [256, 256],
  "train": {
    "manifest": "/path/to/MATLAB_DATA_ROOT/metadata/manifest.csv",
    "augment": true,
    "samples_per_epoch": 1000
  }
}
```

默认模型是 `care2d`。要切换 baseline，只改 `model.name` 和对应参数即可。

```bash
python scripts/train_supervised.py \
  -c configs/supervised_2d.json \
  -o outputs/care2d_supervised
```

例如使用 RCAN2D，把配置中的模型段改成 `configs/model_rcan2d.json` 的内容。

## 自监督训练

自监督训练只需要 MATLAB 生成的 `abe/` 带像差图像目录，不需要无像差标签。建议使用 `scare2d` 或 `picnet2d`，因为它们会输出 Zernike 系数。

修改 [configs/self_supervised_2d.json](configs/self_supervised_2d.json)：

```json
"data": {
  "patch_size": [256, 256],
  "train": {
    "image_dir": "/path/to/MATLAB_DATA_ROOT/abe",
    "augment": true,
    "samples_per_epoch": 1000
  }
}
```

```bash
python scripts/train_self_supervised.py \
  -c configs/self_supervised_2d.json \
  -o outputs/scare2d_self_supervised
```

训练路径：

```text
aberrated image
  -> model -> restored image + zernike coefficients
  -> AO forward model -> estimated aberrated image
  -> loss(estimated aberrated, input aberrated)
```

## 推理

```bash
python scripts/test.py \
  --checkpoint outputs/care2d_supervised/best.pt \
  --input data/ao2d/abe \
  --output outputs/test_results
```

如果模型输出 Zernike 系数，`test.py` 会同时保存 `*_zernike_coeff_um.txt`。

## PICNet 和 SSPAO/SCARE 的区别

这里的 `SSPAO/SCARE` 指原仓库中 `model/SCARE.py` + `train_selfsupervised.py` / `train_ssp_znfigsep.py` 这一类自监督物理一致性路线。

`PICNet` 指原仓库中 `model/PICNet.py` + `train_2_stage.py` / `train_gan.py` 这一类对象生成器、像差生成器分离的路线。

| 对比项 | PICNet | SSPAO/SCARE |
| --- | --- | --- |
| 网络拆分 | 通常拆成 `OBJ_Generator` 和 `Aberration/Phase_Generator`，必要时加 Discriminator | 一个主干网络同时输出 restored object 和 Zernike coefficients |
| 训练方式 | 更偏两阶段/联合训练：先用合成 object + 已知 Zernike 做监督，再用真实 aberrated 图像做物理一致性 | 更直接的自监督：输入 aberrated，输出 restored + coeff，再通过 forward model 重建 aberrated |
| 监督信号 | 可用合成数据提供 object、phase、coeff 的显式监督 | 主要依赖 forward model 的重投影一致性，可加 TV/RSD 等正则 |
| 像差表示 | 原实现里既可预测 Zernike，也可预测 phase map，训练脚本中有 phase loss / coeff loss | 当前 SCARE 主线预测 Zernike coefficient vector |
| 模型复杂度 | 模块更多，训练流程更复杂，但可利用合成监督和 GAN/循环约束 | 结构更简洁，训练脚本更短，更容易作为 SSPAO 主方法复现 |
| 适用场景 | 有可靠 object 仿真、已知/可合成 Zernike 标签，想强化像差估计时更合适 | 只有真实 aberrated 图像，想靠物理模型自监督恢复时更直接 |

一句话：PICNet 更像“分模块、两阶段、可监督/半监督的物理反演框架”；SSPAO/SCARE 更像“单网络端到端自监督 AO 恢复框架”。
