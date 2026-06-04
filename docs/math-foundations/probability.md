# 概率统计

> 概率统计帮助你判断模型中的变化到底是规律、噪声还是偶然。Transformer 训练里的初始化、dropout、mini-batch、采样、softmax 和注意力缩放都需要概率直觉。

## 学习目标

- 理解事件、随机变量、分布、期望、方差、协方差。
- 能解释 softmax 权重为什么不等于“绝对可信概率”。
- 能推导 attention 中 \(1/\sqrt{d_k}\) 缩放因子的直觉。
- 能用代码模拟维度、batch size、随机种子对统计量的影响。

## 从抛硬币开始

抛一次硬币，结果可能是正面或反面。我们可以定义随机变量：

\[
X=
\begin{cases}
1,& \text{正面}\\
0,& \text{反面}
\end{cases}
\]

如果硬币公平：

\[
P(X=1)=0.5,\quad P(X=0)=0.5
\]

期望：

\[
\mathbb{E}[X]=1\cdot 0.5+0\cdot 0.5=0.5
\]

这不是说每次结果都是 0.5，而是长期平均接近 0.5。深度学习实验也类似：一次训练结果不代表稳定结论，要看多次运行或置信区间。

## 随机变量和分布

随机变量是结果不固定的量。分布描述每个结果出现的可能性。

| 对象 | 直觉 | 深度学习例子 |
|---|---|---|
| 随机变量 | 结果不固定 | 初始化权重、采样 token、mini-batch loss |
| 分布 | 可能结果及概率 | softmax 输出、数据分布、噪声分布 |
| 样本 | 一次观察 | 一个 batch 的 loss |
| 估计量 | 用样本估计总体量 | 验证集准确率、平均 loss |

## 期望：长期平均

离散随机变量的期望：

\[
\mathbb{E}[X]=\sum_x xP(X=x)
\]

在训练中，真实目标通常是最小化数据分布上的期望损失：

\[
\mathcal{L}(\theta)=\mathbb{E}_{(x,y)\sim p_{\text{data}}}[\ell(f_\theta(x),y)]
\]

但我们无法每步用完整数据，只能用 mini-batch 估计：

\[
\hat{\mathcal{L}}=\frac{1}{B}\sum_{i=1}^{B}\ell_i
\]

## 方差：波动有多大

方差：

\[
\operatorname{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]
\]

方差大说明结果波动大。训练中常见现象：

- batch 太小，loss 曲线抖动明显。
- 初始化随机性导致不同 seed 结果不同。
- logits 尺度太大，softmax 权重过尖。
- 梯度噪声太大，优化不稳定。

## 协方差：两个量是否一起变化

\[
\operatorname{Cov}(X,Y)=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]
\]

协方差为正：两个量倾向于一起变大或一起变小。协方差为负：一个变大时另一个倾向于变小。

在表示学习中，可以用协方差观察不同特征维是否高度相关。如果很多维度高度相关，表示可能存在冗余。

## softmax 是归一化权重

softmax：

\[
p_i=\frac{\exp(z_i)}{\sum_j \exp(z_j)}
\]

它输出非负且和为 1 的数，因此可以当作分布使用。但它不一定是校准概率。比如模型很自信，不代表一定正确。

```python
import torch

for scale in [1.0, 5.0, 20.0]:
    z = torch.tensor([1.0, 2.0, 3.0]) * scale
    p = torch.softmax(z, dim=0)
    print(scale, p, "max=", p.max().item())
```

scale 越大，softmax 越尖，最大概率越接近 1。

## attention 缩放因子的推导

假设 query 和 key 的每个维度独立，均值为 0，方差为 1：

\[
q_i,k_i \sim \text{mean }0,\ \text{var }1
\]

点积：

\[
q^\top k=\sum_{i=1}^{d_k}q_i k_i
\]

每一项 \(q_i k_i\) 的均值为 0，方差约为 1。若各项近似独立，则：

\[
\operatorname{Var}(q^\top k)=\sum_{i=1}^{d_k}\operatorname{Var}(q_i k_i)\approx d_k
\]

所以 \(d_k\) 越大，logits 波动越大。除以 \(\sqrt{d_k}\) 后：

\[
\operatorname{Var}\left(\frac{q^\top k}{\sqrt{d_k}}\right)\approx 1
\]

这能让 softmax 的输入尺度更稳定，避免一开始就过度饱和。

## 用代码验证缩放

```python
import torch

torch.manual_seed(0)
for d in [16, 64, 256, 1024]:
    q = torch.randn(20000, d)
    k = torch.randn(20000, d)
    dot = (q * k).sum(dim=-1)
    scaled = dot / (d ** 0.5)
    print(d, round(dot.std().item(), 2), round(scaled.std().item(), 2))
```

你应该看到：未缩放点积的标准差随 \(d\) 增大，缩放后的标准差接近 1。

## batch size 与估计方差

样本均值的方差大约随 batch size 变大而降低：

\[
\operatorname{Var}(\bar{X})=\frac{\operatorname{Var}(X)}{B}
\]

代码模拟：

```python
import torch

torch.manual_seed(0)
for B in [4, 16, 64, 256]:
    means = []
    for _ in range(1000):
        x = torch.randn(B)
        means.append(x.mean())
    means = torch.stack(means)
    print(B, round(means.std().item(), 3))
```

batch 越大，均值估计越稳定。但大 batch 也可能改变优化动态，所以不能只看稳定性。

## Transformer 对应关系

| 概率概念 | Transformer 中的对应 |
|---|---|
| 随机变量 | 初始化、dropout、采样 token、mini-batch loss |
| 期望 | 训练目标、泛化误差 |
| 方差 | logits 尺度、梯度噪声、实验波动 |
| 协方差 | 特征相关性、表示冗余 |
| 条件概率 | 语言模型 \(P(x_t\mid x_{<t})\) |
| softmax 分布 | attention 权重、词表预测分布 |

## 常见误区

- 把“期望为 0.5”理解为“每次结果都是 0.5”。
- 单次实验提升就认为方法有效。
- 忽略随机种子和置信区间。
- 把 softmax 输出当成校准概率。
- 推导 attention 缩放时忘记独立性和方差假设。
- 只看平均值，不看方差和极端值。

## 检查问题

1. 随机变量和样本有什么区别？
2. 期望损失和 mini-batch loss 有什么区别？
3. 方差大在训练曲线上会表现为什么？
4. 为什么 \(q^\top k\) 的方差会随 \(d_k\) 增大？
5. softmax 很尖时，梯度和信息路由可能发生什么变化？

## 最小练习

1. 模拟抛硬币 10 次、100 次、10000 次，观察均值变化。
2. 推导 \(q^\top k\) 方差，并写出每一步使用的假设。
3. 比较 `softmax(z)` 和 `softmax(10*z)` 的最大概率。
4. 模拟不同 batch size 下均值估计的标准差。
5. 对一个 `(B,T,D)` 表示计算特征协方差矩阵，并观察是否有高度相关维度。

## 阶段产出

写一页“概率统计到 Transformer 映射表”：至少解释 attention 缩放、softmax 饱和、mini-batch 噪声、随机种子、多次实验平均。每个条目都配一个最小代码观察。
