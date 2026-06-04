# 数学语言与符号入门

> 学数学前，先学会读数学。很多公式卡住不是因为定理太难，而是对象、索引、求和和形状没有读出来。

## 学习目标

- 能区分标量、向量、矩阵、张量、函数、分布和参数。
- 能读懂常见索引：\(x_i\)、\(x^{(i)}\)、\(X_{ij}\)、\(Q_{bhtd}\)。
- 能把求和、均值、期望、范数、argmax 翻译成普通语言。
- 能在读 Transformer 公式时先写 shape，再讨论含义。

## 为什么要先学符号

深度学习论文的公式很少会把每个变量都解释到初学者可读。作者默认你能看出：

- 小写 \(x\) 常表示一个向量或一个样本。
- 大写 \(X\) 常表示矩阵、数据集或随机变量。
- 下标 \(i,j,t\) 常表示位置、维度或样本编号。
- 上标 \((i)\) 常表示第 \(i\) 个样本，不一定是幂。
- \(\sum\) 表示把很多项加起来，常对应代码里的 `.sum()`。

如果这些符号没有读顺，后面的线代、概率、矩阵微分都会变成记忆负担。

## 最小对象表

| 对象 | 例子 | 普通语言 | 代码里常见形状 |
|---|---|---|---|
| 标量 | \(a\) | 一个数 | `()` |
| 向量 | \(x\) | 一排数，表示一个点或一个特征列表 | `(D,)` |
| 矩阵 | \(X\) | 多个向量排成表 | `(T, D)` 或 `(B, D)` |
| 张量 | \(X\) | 带更多轴的数组 | `(B, H, T, D)` |
| 函数 | \(f(x)\) | 输入到输出的规则 | `def f(x): ...` |
| 参数 | \(\theta,W,b\) | 模型要学习的量 | `nn.Parameter` |
| 随机变量 | \(X\) | 结果不固定的量 | 采样得到的 tensor |

一个简单判断：如果它能被一个数字表示，就是标量；如果是一组数字，就是向量；如果是二维表，就是矩阵；如果维度更多，就是张量。

## 下标怎么读

### 向量下标

\[
x = [x_1,x_2,\ldots,x_D]
\]

这里 \(x_i\) 是向量 \(x\) 的第 \(i\) 个分量。若 \(x\) 是一个 token embedding，\(x_i\) 就是第 \(i\) 个 hidden feature。

### 矩阵下标

\[
X =
\begin{bmatrix}
X_{11} & X_{12}\\
X_{21} & X_{22}
\end{bmatrix}
\]

\(X_{ij}\) 表示第 \(i\) 行第 \(j\) 列。若 \(X\in\mathbb{R}^{T\times D}\)，第 \(i\) 行通常表示第 \(i\) 个 token，第 \(j\) 列表示第 \(j\) 个特征。

### 样本上标

\[
x^{(i)}
\]

很多机器学习教材用 \(x^{(i)}\) 表示第 \(i\) 个样本。它不是 \(x\) 的 \(i\) 次方。读到上标时要先判断上下文：是在表示样本编号，还是数学幂。

### Transformer 四维下标

在 attention 代码中，经常有：

\[
Q\in\mathbb{R}^{B\times H\times T\times d_h}
\]

可以读成：`B` 个样本，每个样本 `H` 个 head，每个 head 有 `T` 个 token，每个 token 的 query 向量维度是 \(d_h\)。一个元素 \(Q_{bhtd}\) 表示第 `b` 个样本、第 `h` 个 head、第 `t` 个 token、第 `d` 个特征。

## 求和符号怎么翻译

\[
\sum_{i=1}^{D} x_i
\]

普通语言：把向量 \(x\) 的第 1 个到第 \(D\) 个分量全部加起来。

代码：

```python
x.sum()
```

如果有权重：

\[
\sum_{i=1}^{D} w_i x_i
\]

普通语言：每个 \(x_i\) 乘一个权重 \(w_i\)，再加起来。这就是点积，也是线性层和 attention logits 的基本动作。

```python
(w * x).sum()
```

## 均值、期望、方差

均值是样本层面的平均：

\[
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
\]

期望是随机变量长期平均的理想值：

\[
\mathbb{E}[X]
\]

方差衡量波动：

\[
\operatorname{Var}(X)=\mathbb{E}\left[(X-\mathbb{E}[X])^2\right]
\]

在 Transformer 里，方差非常重要。attention logits 如果方差太大，softmax 会变得很尖；梯度如果方差太大，训练会不稳定。

## 范数是什么

范数可以理解为“向量有多大”。最常见的是 L2 范数：

\[
\|x\|_2=\sqrt{x_1^2+x_2^2+\cdots+x_D^2}
\]

代码：

```python
import torch
x = torch.tensor([3.0, 4.0])
print(torch.linalg.norm(x))  # 5
```

在模型中，范数常用于观察 embedding 是否爆炸、梯度是否过大、表示是否塌缩。

## argmax 与 softmax

\[
\arg\max_i z_i
\]

表示找到最大值所在的位置。它返回的是索引，不是最大值本身。

softmax 把一组分数变成非负且和为 1 的权重：

\[
\operatorname{softmax}(z)_i=\frac{\exp(z_i)}{\sum_j \exp(z_j)}
\]

在 attention 中，softmax 不是“分类概率”这么简单，它是“每个 query 应该从哪些 key/value 位置取信息”的权重。

## 读公式的五步法

遇到公式，先不要急着理解证明，按这个顺序写：

1. 这个式子的输入是什么。
2. 这个式子的输出是什么。
3. 每个变量的 shape 是什么。
4. 每一步操作是加法、乘法、矩阵乘法、归一化还是非线性。
5. 这个式子在模型里对应哪段代码。

例子：

\[
S=\frac{QK^\top}{\sqrt{d_k}}
\]

| 变量 | shape | 含义 |
|---|---|---|
| \(Q\) | `(B, H, T_q, d_k)` | query 向量 |
| \(K\) | `(B, H, T_k, d_k)` | key 向量 |
| \(K^\top\) | `(B, H, d_k, T_k)` | 最后两维转置 |
| \(S\) | `(B, H, T_q, T_k)` | 每个 query 对每个 key 的相似度 |

代码：

```python
import torch
B, H, Tq, Tk, D = 2, 4, 3, 5, 8
Q = torch.randn(B, H, Tq, D)
K = torch.randn(B, H, Tk, D)
S = Q @ K.transpose(-1, -2) / (D ** 0.5)
print(S.shape)  # torch.Size([2, 4, 3, 5])
```

## 常见误区

- 把 \(x_i\) 和 \(x^{(i)}\) 混在一起。
- 看见大写字母就以为一定是矩阵；有时大写也表示随机变量。
- 不写 shape 就开始推导。
- 忽略求和维度，导致 softmax 或 mean 用错轴。
- 认为公式和代码天然一致；实际中 batch/head 维经常被省略。

## 最小练习

1. 写出 \(X\in\mathbb{R}^{B\times T\times D}\) 中 \(X_{btd}\) 的含义。
2. 把 \(\sum_i w_i x_i\) 写成 NumPy 代码。
3. 给定 \(Q,K\in\mathbb{R}^{B\times H\times T\times d}\)，写出 \(QK^\top\) 的输出 shape。
4. 解释 \(\mathbb{E}[X]\) 和 \(\bar{x}\) 的区别。
5. 找一篇 Transformer 论文公式，给每个变量补 shape。

## 阶段产出

完成一页“符号翻译表”，至少包含：下标、上标、求和、均值、期望、方差、范数、softmax、argmax、矩阵乘法。后续每读一篇论文都继续补充。
