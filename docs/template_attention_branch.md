# Template Attention Branch

本文档记录当前 2D Zernike template attention 分支的设计、mask 逻辑和前向计算流程。

## 目标

该分支不是直接从图像纹理回归 Zernike 系数，而是把输入图像转换到频域，再与一组由光学模型生成的 OTF template buffer 做匹配。它主要提供两类信息：

- mode presence：某个 Zernike mode 是否在输入中有对应的频域响应。
- signed response：输入更接近该 mode 的 `+eps` 还是 `-eps` template。

最终系数仍然和主干回归分支融合，template attention 不单独承担全部 mode 的符号和幅值判断。当前配置只让符号可分性较强的 mode 使用 signed template，默认是 `j=6,7,8,9,15`。

## 输入五通道

`OTFTemplateAttentionHead2D._input_features(image)` 固定输出 5 个通道：

1. `log_amp`
   - 对输入图像做中心化 FFT 后取 `log(1 + |Y|)`。
   - 再做每张图内部 z-score。
   - 表示频率能量分布，但会混入 OBJ 纹理和边界结构。

2. `masked cos`
   - `cos(angle(Y))`。
   - 只保留可靠频率区域，低幅值频点置 0。
   - 用于提供频域相位结构。

3. `masked sin`
   - `sin(angle(Y))`。
   - 同样只保留可靠频率区域。
   - 当前 template buffer 里最主要的 signed mode 信息通常集中在该通道。

4. `reliability`
   - `log(1 + |Y|)` 归一化到 `[0, 1]` 后乘以 phase mask。
   - 告诉 encoder 哪些频点的相位可信。
   - 它不是像差信息本身，而是相位通道的置信度提示。

5. `freq radius`
   - 频率半径 `rho`，中心为低频，外侧为高频。
   - 给 encoder 一个固定的频率位置先验。

## Mask 逻辑

当前有两类 mask，分别作用在输入特征和 template buffer。

### 输入 phase mask

位置：`OTFTemplateAttentionHead2D._input_features`

逻辑：

```text
image -> subtract mean -> FFT -> optional fftshift -> spectrum Y
magnitude_score = |Y|
threshold = quantile(magnitude_score, input_phase_mask_percentile)
phase_mask = magnitude_score > max(threshold, eps)
```

默认 `input_phase_mask_percentile = 72.0`，即只保留幅值排名约前 28% 的频点。

原因：

- FFT 幅值很小的位置，相位角接近随机，`cos(angle)` / `sin(angle)` 会看起来像噪声。
- 直接把所有频点的相位给网络，会让网络看到大量没有物理意义的随机相位。
- phase mask 只让网络使用幅值足够高、相位相对可信的区域。

注意：

- `log_amp` 不乘该 mask，因为幅度谱本身在全频域都有意义。
- `masked cos` 和 `masked sin` 乘该 mask。
- `reliability` 也乘该 mask，用于显式标记可信区域。

### Template MTF support mask

位置：`ZernikeOTFTemplateBank2D.rebuild`

逻辑：

```text
psf(+eps), psf(-eps), psf0 -> OTF(+eps), OTF(-eps), OTF0
mtf_support = (|OTF(sign)| > otf_mtf_threshold) & (|OTF0| > otf_mtf_threshold)
relative_phase = OTF(sign) * conj(OTF0)
relative_cos = real(relative_phase) / |relative_phase| * mtf_support
relative_sin = imag(relative_phase) / |relative_phase| * mtf_support
```

默认 `otf_mtf_threshold = 0.03`。

原因：

- 在 OTF 或理想 OTF 幅值很低的位置，相对相位同样不可靠。
- 只在 MTF 支撑区域内使用 relative phase，避免 template 里出现由低幅值除法导致的相位噪声。

## Template Buffer

当前只保留一个实际用于 attention 匹配的 buffer：

```text
templates: (K, 2, 3, H, W)
```

含义：

- `K`：Zernike mode 数量，当前通常为 `j=3..15` 共 13 个。
- `2`：两个符号模板，`+eps` 和 `-eps`。
- `3`：OTF template 通道：
  - `log_amp = log(1 + |OTF(sign)|) - log(1 + |OTF0|)`
  - `relative_cos`
  - `relative_sin`

构造后会对每个 template 做 z-score 和 L2 normalize，方便后续用点积作为相似度。

## 前向逻辑

1. 主模型提取图像空间/梯度/频域分支特征，得到 `features`。

2. attention head 对原始输入图像 `image` 构造五通道频域输入：

```text
input_features = [log_amp, masked_cos, masked_sin, reliability, freq_radius]
```

3. encoder 把五通道输入映射成 3 通道 OTF-like feature：

```text
encoded = encoder(input_features)
encoded = zscore(encoded)
encoded = l2_normalize(encoded)
```

4. 与 template buffer 做匹配：

```text
scores = einsum(encoded, templates)
scores shape = (B, K, 2)

scores_pos = scores[..., 0]
scores_neg = scores[..., 1]
signed_score = scores_pos - scores_neg
best_score = max(scores_pos, scores_neg)
```

5. 得到 template 分支的系数估计：

```text
sign = tanh(response_scale * signed_score)
presence = sigmoid(alpha * (best_score - tau))
magnitude = max_amp_um * presence
a_template = signed_mode ? magnitude * sign : magnitude
```

这里的 `sign` 实际是 `[-1, 1]` 范围内的 signed fraction，不只是离散符号。对于不在 `signed_zernike_indices` 里的 mode，template 分支只输出非负 magnitude，用于 presence/magnitude 辅助，不强行预测符号。

6. 与主干回归结果融合：

```text
a_base = base_head(features)
template_proposal = signed_mode ? a_template : presence * a_base
confidence = confidence_head([scores_pos, scores_neg, best_score, |signed_score|, |a_template|, |template_proposal - a_base|])
a_final = confidence * template_proposal + (1 - confidence) * a_base
```

`template_proposal` 的处理是为了避免对符号不可分 mode 使用错误符号：这类 mode 只用 attention presence 调节主干输出。

## 删除的旧分支

本次整理后删除了以下内容：

- `input_feature_channels == 3` 兼容分支。
  - 当前设计固定使用五通道输入，避免回到未 mask 的相位特征。

- `derivative_templates` buffer。
  - 它本质上是 `(+eps - -eps)` 的额外缓存。
  - 目前前向直接使用 `scores_pos - scores_neg` 作为 signed response。
  - 这样 attention 只依赖一个清晰的 `templates` buffer。

- `eta` 参数。
  - 旧代码中未参与实际前向计算。

- `amplitude_head` / `use_amplitude_head`。
  - 旧代码中未参与实际前向计算。

- 可视化脚本中的 derivative buffer 图、derivative map 和 derivative score。
  - 现在只保留正负 template buffer、正负可分性、positive/negative correlation 和 pos-neg score。

## 当前风险

从 buffer 可视化看，很多 mode 的 `+eps` 和 `-eps` template 高度相似，尤其是部分低阶和高阶 mode。因此该分支更适合作为 mode presence 和频域结构先验；对于符号难分的 mode，仍应依赖主干回归分支和融合置信度。
