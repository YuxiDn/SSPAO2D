# Template Attention Formulas

本文档记录当前 template attention 分支使用到的主要计算公式。对应实现主要位于：

- `src/ao2d/models/zernike_template_attention.py`
- `scripts/pretrain_template_attention.py`

## 1. 输入频域特征

输入图像记为 \(y\)，先做中心化 FFT：

```latex
Y = \operatorname{fftshift}\left(\mathcal{F}_2(y-\bar y)\right)
```

无 reference image 时：

```latex
m = |Y|,\qquad A=\log(1+|Y|),\qquad P=Y
```

有 reference image \(x\) 时：

```latex
X = \operatorname{fftshift}\left(\mathcal{F}_2(x-\bar x)\right)
```

```latex
m = |Y||X|,\qquad
A=\log(1+|Y|)-\log(1+|X|),\qquad
P=Y\overline{X}
```

phase mask：

```latex
M_{\text{phase}}(u,v)=\mathbf{1}\left[m(u,v)>Q_p(m)\right]
```

当前：

```latex
p=72\%
```

幅度 z-score：

```latex
\tilde A =
\frac{A-\mu(A)}
{\sigma(A)+\epsilon}
```

相位 cos/sin：

```latex
C(u,v)=
\frac{\Re(P(u,v))}
{|P(u,v)|+\epsilon}
M_{\text{phase}}(u,v)
```

```latex
S(u,v)=
\frac{\Im(P(u,v))}
{|P(u,v)|+\epsilon}
M_{\text{phase}}(u,v)
```

reliability：

```latex
R(u,v)=
\frac{\log(1+m(u,v))}
{\max_{u,v}\log(1+m(u,v))+\epsilon}
M_{\text{phase}}(u,v)
```

最终五通道输入：

```latex
F_{\text{in}}=
[\tilde A,\ C,\ S,\ R,\ \rho]
```

## 2. Template Buffer

对每个 Zernike mode \(k\)，构造正负扰动：

```latex
a_{k,+}=\epsilon_k e_k,\qquad
a_{k,-}=-\epsilon_k e_k
```

生成 PSF：

```latex
h_{k,q}=\operatorname{PSF}(a_{k,q}),\qquad q\in\{+,-\}
```

```latex
h_0=\operatorname{PSF}(0)
```

转成 OTF：

```latex
H_{k,q}
=
\mathcal{F}_2(\operatorname{ifftshift}(h_{k,q}))
```

```latex
H_0
=
\mathcal{F}_2(\operatorname{ifftshift}(h_0))
```

template 幅度通道：

```latex
T^{\text{amp}}_{k,q}
=
\log(1+|H_{k,q}|)
-
\log(1+|H_0|)
```

relative phase：

```latex
G_{k,q}=H_{k,q}\overline{H_0}
```

MTF support mask：

```latex
M_{\text{MTF}}
=
\mathbf{1}\left[|H_{k,q}|>\tau_{\text{MTF}}\right]
\mathbf{1}\left[|H_0|>\tau_{\text{MTF}}\right]
```

当前：

```latex
\tau_{\text{MTF}}=0.03
```

template 相位通道：

```latex
T^{\cos}_{k,q}
=
\frac{\Re(G_{k,q})}
{|G_{k,q}|+\epsilon}
M_{\text{MTF}}
```

```latex
T^{\sin}_{k,q}
=
\frac{\Im(G_{k,q})}
{|G_{k,q}|+\epsilon}
M_{\text{MTF}}
```

最终 template：

```latex
T_{k,q}
=
\operatorname{L2Norm}
\left(
\operatorname{ZScore}
\left[
T^{\text{amp}}_{k,q},
T^{\cos}_{k,q},
T^{\sin}_{k,q}
\right]
\right)
```

## 3. Attention Score

encoder 输出：

```latex
E
=
\operatorname{L2Norm}
\left(
\operatorname{ZScore}
\left(
f_\theta(F_{\text{in}})
\right)
\right)
```

template 匹配分数：

```latex
s_{b,k,q}
=
\sum_{c,u,v}
E_{b,c,u,v}
T_{k,q,c,u,v}
```

正负分数：

```latex
s^+_{b,k}=s_{b,k,+}
```

```latex
s^-_{b,k}=s_{b,k,-}
```

signed score：

```latex
\Delta s_{b,k}=s^+_{b,k}-s^-_{b,k}
```

best score：

```latex
s^{\max}_{b,k}=\max(s^+_{b,k},s^-_{b,k})
```

## 4. Presence、Sign 和 Template Coefficient

sign：

```latex
\operatorname{sign}_{b,k}
=
\tanh
\left(
\gamma_k \Delta s_{b,k}
\right)
```

presence：

```latex
p_{b,k}
=
\sigma
\left(
\alpha(s^{\max}_{b,k}-\tau_k)
\right)
```

magnitude：

```latex
m_{b,k}=a^{\max}_k p_{b,k}
```

signed mode 的 template coefficient：

```latex
a^{\text{template}}_{b,k}
=
m_{b,k}\operatorname{sign}_{b,k}
```

presence-only mode 的 template coefficient：

```latex
a^{\text{template}}_{b,k}
=
m_{b,k}
```

注意：presence-only mode 中的 \(a^{\text{template}}_{b,k}\) 只是 attention 内部的模板幅值，不作为最终 Zernike 系数直接解释。

## 5. 最终融合

基础回归分支：

```latex
a^{\text{base}}=\operatorname{base\_head}(z)
```

当前 signed mode 集合：

```latex
\mathcal{K}_{\text{signed}}=\{6,7,8,9,15\}
```

template proposal：

```latex
a^{\text{prop}}_{b,k}
=
\begin{cases}
a^{\text{template}}_{b,k},
& k\in \mathcal{K}_{\text{signed}}\\
p_{b,k}a^{\text{base}}_{b,k},
& k\notin \mathcal{K}_{\text{signed}}
\end{cases}
```

confidence 输入：

```latex
g_{b,k}
=
[
s^+_{b,k},
s^-_{b,k},
s^{\max}_{b,k},
|\Delta s_{b,k}|,
|a^{\text{template}}_{b,k}|,
|a^{\text{prop}}_{b,k}-a^{\text{base}}_{b,k}|
]
```

confidence：

```latex
c_{b,k}=\sigma(\operatorname{MLP}(g_{b,k}))
```

最终系数：

```latex
a^{\text{final}}_{b,k}
=
c_{b,k}a^{\text{prop}}_{b,k}
+
(1-c_{b,k})a^{\text{base}}_{b,k}
```

## 6. Pretrain Loss

signed mask：

```latex
M_k=
\mathbf{1}[k\in \mathcal{K}_{\text{signed}}]
```

signed mode coefficient L1：

```latex
\mathcal{L}_{\text{signed}}
=
\lambda_1
\frac{1}{|\mathcal{K}_{\text{signed}}|}
\sum_{k\in\mathcal{K}_{\text{signed}}}
|a^{\text{pred}}_k-a^{\text{gt}}_k|
```

如果启用 MSE：

```latex
\mathcal{L}_{\text{mse}}
=
\lambda_2
\frac{1}{|\mathcal{K}_{\text{signed}}|}
\sum_{k\in\mathcal{K}_{\text{signed}}}
(a^{\text{pred}}_k-a^{\text{gt}}_k)^2
```

当前 presence-only magnitude loss 关闭：

```latex
\lambda_{\text{mag}}=0
```

因此 presence-only mode 当前不使用：

```latex
\left||a^{\text{pred}}_k|-|a^{\text{gt}}_k|\right|
```

active label：

```latex
y^{\text{active}}_{b,k}
=
\mathbf{1}
\left[
|a^{\text{gt}}_{b,k}|>\delta
\right]
```

当前：

```latex
\delta=0.02\ \mu m
```

presence BCE：

```latex
\mathcal{L}_{\text{presence}}
=
\lambda_p
\operatorname{BCE}
(p_{b,k},y^{\text{active}}_{b,k})
```

当前核心 loss：

```latex
\mathcal{L}
=
\mathcal{L}_{\text{signed}}
+
\mathcal{L}_{\text{mse}}
+
\mathcal{L}_{\text{presence}}
```

其中 presence-only mode 不再用 template 幅值直接拟合真实系数幅值。
