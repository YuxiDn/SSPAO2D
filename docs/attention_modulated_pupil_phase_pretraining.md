# Attention-Modulated Pupil Phase Pretraining

本文档记录当前 `attention_modulated_pupil_phase` 预训练框架。核心思想是：

$$
a_{\mathrm{final},k}
=
r_k a_{\mathrm{base},k}
$$

其中：

$$
r_k = 1 + \lambda(g_k-\bar g)
$$

也就是说，pupil phase projection 分支负责预测有符号、有幅值的 Zernike 系数；template attention 分支只负责判断哪些 mode 更强或更弱，然后调制物理分支输出。

## 1. 总体结构

输入带像差图像记为：

$$
y
$$

synthetic 训练时，带像差图像由 clean object 和随机 Zernike 系数生成：

$$
y = h(a_{\mathrm{gt}}) * x
$$

其中：

- \(x\)：clean object。
- \(a_{\mathrm{gt}}\)：真实采样的 Zernike 系数。
- \(h(a_{\mathrm{gt}})\)：由 Zernike 系数生成的 PSF。
- \(*\)：卷积。
- \(y\)：带像差图像。

网络分两条路：

$$
y \rightarrow F_y
$$

物理系数路：

$$
F_y
\rightarrow
\text{pupil phase projection branch}
\rightarrow
a_{\mathrm{base}}
$$

attention 诊断路：

$$
y
\rightarrow
\text{template attention branch}
\rightarrow
g_k
$$

最后：

$$
a_{\mathrm{final},k}
=
r_k a_{\mathrm{base},k}
$$

## 2. Pupil Phase Projection 分支

这条分支负责预测一个 pupil OPD map：

$$
\hat\phi(u,v)
$$

这里更准确地说是 OPD map，单位可以理解为 \(\mu m\) 级别的波前高度，而不是 wrapped phase。

只保留 pupil 区域：

$$
M_{\mathrm{pupil}}(u,v)
=
\mathbf{1}[\rho(u,v)\le 1]
$$

去 piston：

$$
\hat \phi
\leftarrow
\left(
\hat \phi
-
\frac{
\sum_{u,v}\hat \phi(u,v)M_{\mathrm{pupil}}(u,v)
}{
\sum_{u,v}M_{\mathrm{pupil}}(u,v)
}
\right)
M_{\mathrm{pupil}}
$$

参数含义：

- \(\hat \phi(u,v)\)：网络预测的 pupil OPD map。
- \(u,v\)：pupil plane 坐标。
- \(\rho(u,v)\)：归一化 pupil 半径。
- \(M_{\mathrm{pupil}}\)：pupil mask。
- piston：整体常数相位，对成像 PSF 没有实际影响，所以去掉。

然后通过固定 Zernike projection 得到：

$$
a_{\mathrm{base}} = \Pi_Z(\hat \phi)
$$

当前实现采用 pairwise phase difference projection。先采样 pupil 内点对：

$$
(u_i^{(1)},v_i^{(1)}),
\quad
(u_i^{(2)},v_i^{(2)})
$$

预测相位差：

$$
\Delta \hat \phi_i
=
\hat \phi(u_i^{(1)},v_i^{(1)})
-
\hat \phi(u_i^{(2)},v_i^{(2)})
$$

每个 Zernike mode 的差分基：

$$
D_{i,k}
=
Z_k(u_i^{(1)},v_i^{(1)})
-
Z_k(u_i^{(2)},v_i^{(2)})
$$

于是：

$$
\Delta \hat\phi \approx D a
$$

固定 ridge 最小二乘投影：

$$
a_{\mathrm{base}}
=
(D^\top D + \eta I)^{-1}D^\top \Delta\hat\phi
$$

参数含义：

- \(D\)：Zernike pairwise difference basis。
- \(a_{\mathrm{base}}\)：物理分支输出的 Zernike 系数。
- \(\eta\)：ridge 正则，对应 `delta_phi_ridge`。
- \(I\)：单位矩阵。
- `delta_phi_pair_count`：采样多少个 pupil 点对。
- `delta_phi_pupil_grid_size`：pupil phase map 的网格大小。
- `delta_phi_max_opd`：限制预测 OPD 的最大范围，避免发散。

这条分支承担：

$$
\text{signed coefficient：正负和幅值}
$$

## 3. Template Attention 分支

attention 分支不直接预测 signed coefficient。它的任务是判断：

$$
\text{当前图像更像哪些 Zernike mode 的频域响应}
$$

输入图像做 FFT：

$$
Y=\operatorname{fftshift}(\mathcal F_2(y-\bar y))
$$

构造五通道输入：

$$
F_{\mathrm{in}}
=
[\tilde A,\ C,\ S,\ R,\rho]
$$

其中：

$$
\tilde A
=
\frac{
\log(1+|Y|)-\mu
}{
\sigma+\epsilon
}
$$

$$
C=
\cos(\angle Y)M_{\mathrm{phase}}
$$

$$
S=
\sin(\angle Y)M_{\mathrm{phase}}
$$

$$
R=
\frac{\log(1+|Y|)}
{\max\log(1+|Y|)+\epsilon}
M_{\mathrm{phase}}
$$

参数含义：

- \(\tilde A\)：归一化 FFT log amplitude。
- \(C\)：可靠频点上的 phase cosine。
- \(S\)：可靠频点上的 phase sine。
- \(R\)：相位可靠性图。
- \(\rho\)：频率半径先验。
- \(M_{\mathrm{phase}}\)：只保留幅值较高的频点，避免低幅值相位噪声。
- `input_phase_mask_percentile=72`：只保留幅值排名前约 28% 的频点用于相位。

encoder 输出：

$$
E=f_\theta(F_{\mathrm{in}})
$$

再做 z-score 和 L2 normalize：

$$
\hat E
=
\operatorname{L2Norm}
(
\operatorname{ZScore}(E)
)
$$

每个 mode 有正负 template：

$$
T_{k,+},\quad T_{k,-}
$$

attention score：

$$
s_{k,q}
=
\langle \hat E, T_{k,q}\rangle
$$

其中：

$$
q\in\{+,-\}
$$

取最大响应：

$$
s_k^{\max}
=
\max(s_{k,+},s_{k,-})
$$

这一步只问该 mode 的无符号证据强不强，不使用：

$$
s_{k,+}-s_{k,-}
$$

去判断符号，因为前期结果显示 signed template 的正负可分性不稳定。

mode strength distribution：

$$
g_k
=
\operatorname{softmax}
(
\beta s_k^{\max}
)
$$

参数含义：

- \(s_{k,+}\)：当前图像和第 \(k\) 个 mode 正向 template 的相似度。
- \(s_{k,-}\)：当前图像和第 \(k\) 个 mode 负向 template 的相似度。
- \(s_k^{\max}\)：第 \(k\) 个 mode 的无符号证据强度。
- \(g_k\)：attention 认为第 \(k\) 个 mode 的相对强度。
- \(\beta\)：temperature，对应 `attention_temperature`。
- \(\beta\) 越大，\(g_k\) 越尖锐；越小，\(g_k\) 越平滑。

## 4. Attention 调制

当前使用 centered modulation：

$$
a_{\mathrm{final},k}
=
r_k a_{\mathrm{base},k}
$$

其中：

$$
r_k
=
1+\lambda(g_k-\bar g)
$$

$$
\bar g
=
\frac{1}{K}\sum_{j=1}^{K}g_j
$$

参数含义：

- \(a_{\mathrm{base},k}\)：pupil phase projection 分支输出的第 \(k\) 阶 Zernike 系数。
- \(a_{\mathrm{final},k}\)：最终用于 loss 和 forward model 的系数。
- \(g_k\)：attention 给出的第 \(k\) 阶相对强度。
- \(\bar g\)：所有 mode 的平均 attention 强度。
- \(r_k\)：第 \(k\) 阶的调制因子。
- \(\lambda\)：调制幅度，对应 `modulation_lambda`。
- \(K\)：Zernike mode 数量，目前是 13。

使用 centered modulation 的原因：

$$
\sum_k g_k=1
$$

如果直接使用：

$$
r_k=1+\lambda g_k
$$

所有 mode 都只会被增强。centered 后：

$$
g_k>\bar g
\Rightarrow
r_k>1
$$

$$
g_k<\bar g
\Rightarrow
r_k<1
$$

因此它能表达强 mode 增强、弱 mode 压低。

## 5. 为什么不用 Convex Fusion

一个可选公式是：

$$
a_{\mathrm{final},k}
=
c_k a_{\mathrm{template},k}
+
(1-c_k)a_{\mathrm{base},k}
$$

这个公式要求：

$$
a_{\mathrm{template},k}
$$

本身是可信 signed coefficient。但前期结果显示：

$$
\operatorname{sign}_k \approx 0
$$

因此：

$$
a_{\mathrm{template},k}
=
a_k^{\max}p_k\operatorname{sign}_k
\approx 0
$$

如果现在使用 convex fusion，可能会把更可靠的 \(a_{\mathrm{base}}\) 拉向 0。因此当前采用：

$$
a_{\mathrm{final},k}
=
r_k a_{\mathrm{base},k}
$$

这样符号和主体幅值都来自物理分支，attention 只负责增强或削弱。

## 6. Mode Strength 监督

旧的 binary presence target 是：

$$
y_k=\mathbf{1}(|a_k|>\delta)
$$

对应 BCE：

$$
L_{\mathrm{BCE}}
=
-
\sum_k
[
y_k\log p_k+(1-y_k)\log(1-p_k)
]
$$

该目标只判断 mode 是否超过阈值，不适合所有 mode 基本都存在、但强弱不同的设置。

当前使用相对强度分布：

$$
w_k
=
\frac{|a_{\mathrm{gt},k}|+\epsilon}
{\sum_j(|a_{\mathrm{gt},j}|+\epsilon)}
$$

其中：

- \(w_k\)：真实第 \(k\) 阶 mode 的相对强度。
- \(a_{\mathrm{gt},k}\)：真实 Zernike 系数。
- \(\epsilon\)：防止除零。

attention 输出：

$$
g_k
=
\operatorname{softmax}(\beta s_k^{\max})
$$

用 KL loss：

$$
L_{\mathrm{strength}}
=
D_{\mathrm{KL}}(w\|g)
=
\sum_k
w_k
\log
\frac{w_k}{g_k+\epsilon}
$$

作用是让：

$$
g_k \approx w_k
$$

也就是让 attention 学会哪些 Zernike mode 在当前样本里贡献更大。

## 7. Pupil Phase 监督

synthetic 训练知道真实系数：

$$
a_{\mathrm{gt}}
$$

因此可以构造真实 OPD：

$$
\phi_{\mathrm{gt}}(u,v)
=
\sum_k
a_{\mathrm{gt},k}Z_k(u,v)
$$

监督预测 phase：

$$
L_{\phi}
=
\left\|
\hat\phi-\phi_{\mathrm{gt}}
\right\|_{1,\mathrm{pupil}}
$$

参数含义：

- \(\phi_{\mathrm{gt}}\)：由真实 Zernike 系数合成的 pupil OPD。
- \(\hat\phi\)：网络预测的 pupil OPD。
- \(Z_k\)：第 \(k\) 个 Zernike basis。
- \(L_{\phi}\)：让 projection 分支学出可解释 pupil phase。
- `pupil_phase_l1_weight`：该 loss 的权重。

## 8. 系数监督

最终系数监督：

$$
L_a
=
\|a_{\mathrm{final}}-a_{\mathrm{gt}}\|_1
$$

如果启用 MSE：

$$
L_{\mathrm{mse}}
=
\|a_{\mathrm{final}}-a_{\mathrm{gt}}\|_2^2
$$

参数含义：

- \(a_{\mathrm{final}}\)：attention 调制后的最终系数。
- \(a_{\mathrm{gt}}\)：真实 Zernike 系数。
- `coeff_l1_weight`：L1 系数 loss 权重。
- `coeff_mse_weight`：MSE 系数 loss 权重。

## 9. 重投影监督

用最终系数重新生成图像：

$$
\hat y
=
h(a_{\mathrm{final}})*x
$$

和输入 synthetic aberrated image 对齐：

$$
L_{\mathrm{reproj}}
=
\|\hat y-y\|_1
$$

参数含义：

- \(\hat y\)：用预测系数重投影得到的带像差图像。
- \(y\)：真实 synthetic 带像差图像。
- \(x\)：clean object。
- \(h(a_{\mathrm{final}})\)：预测系数对应 PSF。
- `reprojection_l1_weight`：重投影 loss 权重。

## 10. 总 Loss

当前整体目标：

$$
L
=
\lambda_a L_a
+
\lambda_{\mathrm{mse}}L_{\mathrm{mse}}
+
\lambda_{\phi}L_{\phi}
+
\lambda_{\mathrm{strength}}L_{\mathrm{strength}}
+
\lambda_{\mathrm{reproj}}L_{\mathrm{reproj}}
$$

当前配置：

```json
"coeff_l1_weight": 1.0,
"coeff_mse_weight": 0.1,
"mode_strength_kl_weight": 0.2,
"pupil_phase_l1_weight": 1.0,
"reprojection_l1_weight": 0.1,
"presence_bce_weight": 0.0,
"otf_feature_cosine_weight": 0.0
```

含义：

- `coeff_l1_weight=1.0`：主要优化最终 Zernike 系数。
- `coeff_mse_weight=0.1`：轻量惩罚大误差。
- `mode_strength_kl_weight=0.2`：让 attention 学 mode 相对强弱。
- `pupil_phase_l1_weight=1.0`：让物理分支学出可解释 pupil phase。
- `reprojection_l1_weight=0.1`：让最终系数通过 forward model 解释图像。
- `presence_bce_weight=0.0`：关闭旧二值 presence。
- `otf_feature_cosine_weight=0.0`：暂时不强迫 encoder 拟合 OTF feature，避免和 strength attention 目标混在一起。

## 11. Batch 构造

当前 synthetic 设置：

```json
"batch_size": 128,
"objects_per_synthetic_batch": 8,
"synthetic_batches_per_epoch": 250
```

每个 step 取：

$$
8\ \text{objects}
$$

每个 object 配多组随机 Zernike：

$$
128 / 8 = 16
$$

所以：

$$
B=128
$$

这样避免一个 batch 只有一个 object，降低网络记住 OBJ 纹理的风险。

## 12. 关键配置参数

`aberration_head_type = attention_modulated_pupil_phase`

使用新结构：

$$
\text{pupil phase projection} + \text{template attention modulation}
$$

`epsilon_um = 0.05`

构造 template bank 时每个 mode 的微小扰动：

$$
a_{k,+}=+\epsilon e_k,\quad a_{k,-}=-\epsilon e_k
$$

`encoder_type = depthwise`

attention branch 的频域 encoder 使用 depthwise separable block。

`encoder_channels = 32`

attention encoder 的隐藏通道数。

`encoder_blocks = 4`

attention encoder 的 block 数量。

`encoder_depthwise_kernel = 5`

depthwise 卷积核大小。

`input_phase_mask_percentile = 72`

相位特征只保留 FFT 幅值排名前约 28% 的频点。

`otf_mtf_threshold = 0.03`

template relative phase 只在 MTF 支撑区域内计算，低 OTF 幅值位置不信任相位。

`modulation_lambda = 0.5`

调制强度：

$$
r_k=1+\lambda(g_k-\bar g)
$$

\(\lambda\) 越大，attention 对 \(a_{\mathrm{base}}\) 的影响越强。

`attention_temperature = 8.0`

softmax 温度：

$$
g_k=\operatorname{softmax}(\beta s_k^{\max})
$$

\(\beta\) 越大，attention 分布越尖锐。

`centered_modulation = true`

使用：

$$
g_k-\bar g
$$

而不是直接使用 \(g_k\)。这样可以同时增强强 mode、压低弱 mode。

`delta_phi_pair_count`

pupil phase projection 使用的 pairwise phase differences 数量。

`delta_phi_pupil_grid_size`

pupil phase map 的空间分辨率。

`delta_phi_ridge`

Zernike projection 的 ridge 正则：

$$
(D^\top D+\eta I)^{-1}D^\top
$$

`delta_phi_max_opd`

限制预测 OPD 的最大值。

## 13. 主要评估指标

旧指标 `presence_acc` 不再是核心，因为 BCE presence 已关闭。

当前重点看：

1. 最终系数误差：

$$
val\_mae,\quad val\_rmse
$$

2. 物理分支自身误差：

$$
base\_mae
$$

如果：

$$
base\_mae < final\_mae
$$

说明 attention modulation 可能拉坏了 projection 分支。

3. attention 强弱分布：

$$
strength\_kl
$$

越低越好。

4. 最强 mode 是否对：

$$
strength\_top1
$$

5. attention 分布和真实强度分布相关性：

$$
strength\_corr
$$

越高越好。

6. 调制是否过强：

$$
modulation\_mean,\quad modulation\_std
$$

如果 `modulation_std` 太大，说明 attention gate 可能过度改变系数。

## 14. 总结

当前训练目标可以概括为：

$$
\boxed{
\text{pupil phase projection 学 signed coefficient，attention 学 mode strength，再用 mode strength 调制 coefficient}
}
$$

完整路径：

$$
y
\rightarrow
\hat\phi
\rightarrow
a_{\mathrm{base}}
$$

$$
y
\rightarrow
g_k
$$

$$
a_{\mathrm{final},k}
=
\left[
1+\lambda(g_k-\bar g)
\right]
a_{\mathrm{base},k}
$$

两条分支的职责不同：

- 物理分支回答：每个 mode 的正负和幅值是多少。
- attention 分支回答：哪些 mode 在当前图像里更强、更值得强调。
