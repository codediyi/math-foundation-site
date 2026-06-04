# 线性代数

> 线性代数不是一堆矩阵公式，而是理解“表示如何被变换、组合、压缩”的语言。Transformer 的 embedding、线性层、Q/K/V 投影、attention logits、低秩压缩都离不开它。

## 学习目标

- 从标量、向量、矩阵开始，理解矩阵乘法的真实含义。
- 能用 shape 解释 embedding、线性层、Q/K/V 投影和 attention logits。
- 掌握向量空间、基、线性组合、秩、投影、特征值、SVD 的基本直觉。
- 能写一个低秩近似实验，并解释奇异值曲线说明了什么。

## 先从一个推荐系统/语言模型例子开始

假设一个 token 被表示成 4 维向量：

\[
x=[0.2,\,-1.1,\,0.7,\,0.4]
\]

这 4 个数本身不一定有人工可解释含义，但它们共同表示这个 token 在模型空间中的位置。线性层做的是：

\[
y=xW+b
\]

如果 \(x\in\mathbb{R}^{1\times 4}\)，\(W\in\mathbb{R}^{4\times 3}\)，那么 \(y\in\mathbb{R}^{1\times 3}\)。普通语言：把 4 维表示重新组合成 3 维表示。

```python
import torch

x = torch.tensor([[0.2, -1.1, 0.7, 0.4]])  # (1, 4)
W = torch.randn(4, 3)
b = torch.randn(3)
y = x @ W + b
print(y.shape)  # (1, 3)
```

线性代数的第一件事，就是看到 `x @ W` 时知道它不是神秘操作，而是多个加权求和。

## 标量、向量、矩阵

| 对象 | 形状 | 直觉 | Transformer 例子 |
|---|---|---|---|
| 标量 | `()` | 一个数 | loss、学习率、一个 logits 值 |
| 向量 | `(D,)` | 一个点或一组特征 | 一个 token embedding |
| 矩阵 | `(T,D)` | 多个向量排成表 | 一段序列的 hidden states |
| 三维张量 | `(B,T,D)` | 多个样本的序列表示 | batch 输入 |
| 四维张量 | `(B,H,T,d_h)` | 多头注意力表示 | 拆 head 后的 Q/K/V |

本章重点讲向量和矩阵；张量只是把矩阵计算批量化。

## 向量：一个点，也是一组特征

向量可以理解为坐标：

\[
x=[x_1,x_2,\ldots,x_D]
\]

也可以理解为特征列表。两个向量是否相似，常用点积：

\[
x^\top y=\sum_{i=1}^{D}x_i y_i
\]

如果两个向量方向接近，点积通常较大；如果方向相反，点积可能为负。

```python
x = torch.tensor([1.0, 2.0, 0.0])
y = torch.tensor([2.0, 1.0, 0.0])
print(torch.dot(x, y))  # 4
```

attention logits 本质上就是 query 和 key 的点积。

## 矩阵乘法：很多个点积

矩阵乘法：

\[
C=AB,\quad C_{ij}=\sum_k A_{ik}B_{kj}
\]

普通语言：\(C\) 的第 \(i,j\) 个元素，是 \(A\) 的第 \(i\) 行和 \(B\) 的第 \(j\) 列做点积。

如果：

\[
X\in\mathbb{R}^{T\times D},\quad W\in\mathbb{R}^{D\times M}
\]

那么：

\[
XW\in\mathbb{R}^{T\times M}
\]

它表示对每个 token 的 \(D\) 维向量做同一个线性变换，得到 \(M\) 维向量。

## 线性组合、基和子空间

线性组合：

\[
a_1v_1+a_2v_2+\cdots+a_kv_k
\]

意思是用若干基础方向 \(v_i\)，按权重 \(a_i\) 拼出一个新向量。

如果一组向量能拼出某个空间中的所有向量，它们就像这个空间的“坐标轴”。这组向量可以称为基。深度学习里的表示空间不一定有清晰人工语义，但线性层经常可以看作“换一组方向重新表达信息”。

## 秩：信息维度有多少

矩阵的秩可以粗略理解为：矩阵里真正独立的方向有多少。

如果一个矩阵的很多行/列都可以由少数方向组合出来，它就是低秩的。低秩意味着信息有冗余，也意味着可以压缩。

```python
import torch

U = torch.randn(20, 3)
V = torch.randn(3, 20)
A = U @ V
print(torch.linalg.matrix_rank(A))  # 通常不超过 3
```

这和低秩 attention、KV cache 压缩有关：如果 attention map 或 hidden states 实际上集中在少数方向上，就可能用更少维度近似。

## 投影：把向量落到某个方向或子空间

把向量 \(x\) 投影到单位方向 \(u\) 上：

\[
\operatorname{proj}_u(x)=(x^\top u)u
\]

直觉：先用点积算 \(x\) 在方向 \(u\) 上有多少分量，再乘回 \(u\)。

```python
x = torch.tensor([3.0, 4.0])
u = torch.tensor([1.0, 0.0])
proj = torch.dot(x, u) * u
print(proj)  # tensor([3., 0.])
```

在表示学习中，投影帮助你理解“某个方向是否承载了某种信息”。

## 特征值和特征向量

如果：

\[
Av=\lambda v
\]

那么 \(v\) 是特征向量，\(\lambda\) 是特征值。直觉：矩阵 \(A\) 作用在方向 \(v\) 上时，只拉伸或压缩，不改变方向。

特征值常用于分析变换是否放大某些方向。若某些方向被过度放大，可能导致数值不稳定或梯度异常。

## SVD：任何矩阵都可以拆成方向、强度、方向

奇异值分解：

\[
A=U\Sigma V^\top
\]

可以读成：

1. \(V^\top\)：先把输入转到一组方向上。
2. \(\Sigma\)：按奇异值缩放每个方向。
3. \(U\)：再转到输出空间。

截断 SVD：

\[
A_k=U_k\Sigma_kV_k^\top
\]

只保留最大的 \(k\) 个奇异值，得到 rank-\(k\) 近似。若误差很小，说明矩阵主要信息集中在少数方向。

## Transformer 对应关系

| 线代概念 | Transformer 中的对应 |
|---|---|
| 向量 | token embedding、query、key、value |
| 点积 | query-key 相似度 |
| 矩阵乘法 | 线性层、attention logits、weighted sum |
| 线性变换 | Q/K/V/O 投影、FFN 第一层和第二层 |
| 秩 | head 冗余、低秩结构、表示压缩 |
| 投影 | 子空间选择、特征方向解释 |
| SVD | 低秩近似、谱分析、KV cache 压缩 |

## 逐步例题：从 Q/K 到 attention logits

假设单个 head：

\[
Q,K\in\mathbb{R}^{T\times d}
\]

logits：

\[
S=QK^\top
\]

输出 shape：

\[
S\in\mathbb{R}^{T\times T}
\]

第 \(i,j\) 个元素：

\[
S_{ij}=q_i^\top k_j
\]

普通语言：第 \(i\) 个 query 对第 \(j\) 个 key 的相似度。

```python
T, d = 4, 8
Q = torch.randn(T, d)
K = torch.randn(T, d)
S = Q @ K.T
print(S.shape)  # (4, 4)
```

如果扩展到 batch 和 head：

```python
B, H, T, d = 2, 3, 4, 8
Q = torch.randn(B, H, T, d)
K = torch.randn(B, H, T, d)
S = Q @ K.transpose(-1, -2)
print(S.shape)  # (2, 3, 4, 4)
```

## 低秩近似实验

```python
import torch

torch.manual_seed(0)
A = torch.randn(64, 64)
U, S, Vh = torch.linalg.svd(A, full_matrices=False)

for k in [4, 8, 16, 32, 64]:
    Ak = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
    err = torch.linalg.norm(A - Ak) / torch.linalg.norm(A)
    print(k, round(err.item(), 4))
```

观察点：

- \(k\) 越大，近似误差越小。
- 如果一个真实 attention map 的误差下降很快，说明它可能有低秩结构。
- 如果下降很慢，强行低秩压缩可能损失信息。

## 常见误区

- 认为矩阵乘法只是“套公式”，没有看成很多个点积。
- 忘记 \(AB\) 和 \(BA\) 通常不相等。
- 把特征分解和 SVD 混为一谈；SVD 适用于更一般的矩阵。
- 只看矩阵 shape，不问每一轴语义。
- 看到低秩就以为一定能压缩；还要看任务指标和误差。

## 检查问题

1. \(XW\) 中 \(X\)、\(W\)、输出分别是什么 shape？
2. attention logits 的第 \(i,j\) 个元素表示什么？
3. 秩低意味着什么信息结构？
4. SVD 中奇异值衰减快说明什么？
5. 如果一个 head 的 attention map 近似低秩，可能带来什么压缩机会？

## 最小练习

1. 用代码验证 `(T,D) @ (D,M) = (T,M)`。
2. 写出 \(QK^\top\) 在四维张量下的 shape。
3. 构造一个 rank-2 矩阵，并验证它的矩阵秩。
4. 对随机矩阵和低秩矩阵分别画奇异值曲线。
5. 找一篇低秩 attention 论文，标出它压缩的是哪个矩阵。

## 阶段产出

写一页“线性代数到 Transformer 映射表”，至少包含：embedding、linear、Q/K/V、attention logits、低秩、SVD、投影。每个条目都要写 shape 和一句直觉。
