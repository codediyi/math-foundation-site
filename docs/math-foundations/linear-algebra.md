# 线性代数

> 线性代数不是一堆矩阵公式，而是理解“表示如何被变换、组合、压缩”的语言。Transformer 的 embedding、线性层、Q/K/V 投影、attention logits、低秩压缩都离不开它。

## 学习目标

- 从标量、向量、矩阵开始，理解矩阵乘法的真实含义。
- 能用 shape 解释 embedding、线性层、Q/K/V 投影和 attention logits。
- 掌握向量空间、基、线性组合、秩、投影、特征值、SVD 的基本直觉。
- 能写一个低秩近似实验，并解释奇异值曲线说明了什么。
- 能把线性代数概念迁移到 Transformer 表示分析和结构压缩。

## 从一个语言模型例子开始

一个 token 被 embedding 成 4 维向量：

\[
x=[0.2,-1.1,0.7,0.4].
\]

线性层：

\[
y=xW+b.
\]

若 \(x\in\mathbb{R}^{1\times 4}\)，\(W\in\mathbb{R}^{4\times 3}\)，输出 \(y\in\mathbb{R}^{1\times 3}\)。普通语言：把 4 维表示重新组合成 3 维表示。

```python
import torch

x = torch.tensor([[0.2, -1.1, 0.7, 0.4]])
W = torch.randn(4, 3)
b = torch.randn(3)
y = x @ W + b
print(y.shape)
```

看到 `x @ W` 时，要想到：这是多个加权求和，不是黑盒。

## 标量、向量、矩阵、张量

| 对象 | shape | 直觉 | Transformer 例子 |
|---|---|---|---|
| 标量 | `()` | 一个数 | loss、学习率、单个 logit |
| 向量 | `(D,)` | 一个点或特征列表 | token embedding、query |
| 矩阵 | `(T,D)` | 多个向量排成表 | 一个序列的 hidden states |
| 三维张量 | `(B,T,D)` | 多个样本的序列表示 | batch hidden states |
| 四维张量 | `(B,H,T,d_h)` | 多头表示 | split head 后的 Q/K/V |

## 向量：方向、长度和相似度

向量：

\[
x=[x_1,x_2,\ldots,x_D].
\]

L2 范数：

\[
\|x\|_2=\sqrt{\sum_i x_i^2}.
\]

点积：

\[
x^\top y=\sum_i x_i y_i.
\]

点积越大，通常说明两个向量方向越接近或范数越大。attention logits 就是 query 与 key 的点积。

```python
x = torch.tensor([1.0, 2.0, 0.0])
y = torch.tensor([2.0, 1.0, 0.0])
print(torch.dot(x, y))
print(torch.linalg.norm(x))
```

## 矩阵乘法：很多个点积

矩阵乘法：

\[
C=AB,
\quad
C_{ij}=\sum_k A_{ik}B_{kj}.
\]

如果：

\[
X\in\mathbb{R}^{T\times D},\quad W\in\mathbb{R}^{D\times M},
\]

那么：

\[
XW\in\mathbb{R}^{T\times M}.
\]

每个 token 的 \(D\) 维表示都被同一个 \(W\) 映射到 \(M\) 维。

## 线性组合、基和子空间

线性组合：

\[
a_1v_1+a_2v_2+\cdots+a_kv_k.
\]

直觉：用若干基础方向拼出新向量。若一组向量能表示空间中的所有向量，它们构成一组基。

在模型里，线性层可以看成把表示投影到新的方向组合上。不同 head 也可以理解为在不同子空间里计算相似度。

## 秩：真正独立的方向有多少

矩阵的秩表示独立方向数量。低秩意味着信息有冗余，可能可以压缩。

```python
U = torch.randn(20, 3)
V = torch.randn(3, 20)
A = U @ V
print(torch.linalg.matrix_rank(A))
```

若 attention map 或 hidden states 有低秩结构，就可能支持低秩 attention 或 KV cache 压缩。

## 投影：保留某个方向上的分量

投影到单位向量 \(u\)：

\[
\operatorname{proj}_u(x)=(x^\top u)u.
\]

```python
x = torch.tensor([3.0, 4.0])
u = torch.tensor([1.0, 0.0])
proj = torch.dot(x, u) * u
print(proj)
```

在机制解释中，投影可用于观察某个方向是否承载特定信息。

## 特征值和特征向量

若：

\[
Av=\lambda v,
\]

则 \(v\) 是特征向量，\(\lambda\) 是特征值。矩阵沿这个方向只拉伸或压缩，不改变方向。

特征值可帮助分析某些方向是否被过度放大，从而影响稳定性。

## SVD：方向、强度、方向

奇异值分解：

\[
A=U\Sigma V^\top.
\]

读法：

1. \(V^\top\)：把输入转到一组方向上。
2. \(\Sigma\)：按奇异值缩放每个方向。
3. \(U\)：转到输出空间。

截断 SVD：

\[
A_k=U_k\Sigma_kV_k^\top.
\]

只保留最大的 \(k\) 个奇异值，得到 rank-\(k\) 近似。

## attention logits 的线代解释

单个 head：

\[
Q,K\in\mathbb{R}^{T\times d},
\quad
S=QK^\top\in\mathbb{R}^{T\times T}.
\]

第 \(i,j\) 个元素：

\[
S_{ij}=q_i^\top k_j.
\]

普通语言：第 \(i\) 个 query 对第 \(j\) 个 key 的相似度。

```python
T, d = 4, 8
Q = torch.randn(T, d)
K = torch.randn(T, d)
S = Q @ K.T
print(S.shape)
```

四维版本：

```python
B, H, T, d = 2, 3, 4, 8
Q = torch.randn(B, H, T, d)
K = torch.randn(B, H, T, d)
S = Q @ K.transpose(-1, -2)
print(S.shape)
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

如果奇异值衰减快，小 \(k\) 也能得到低误差；如果衰减慢，低秩压缩会损失较多信息。

## Transformer 对应关系

| 线代概念 | Transformer 中的对应 |
|---|---|
| 向量 | token embedding、query、key、value |
| 点积 | query-key 相似度 |
| 矩阵乘法 | 线性层、attention logits、weighted sum |
| 线性变换 | Q/K/V/O 投影、FFN |
| 秩 | head 冗余、低秩结构、压缩潜力 |
| 投影 | 子空间选择、特征方向解释 |
| SVD | 低秩近似、谱分析、KV cache 压缩 |

## 常见误区

- 认为矩阵乘法只是套公式，不看成很多个点积。
- 忘记 \(AB\) 和 \(BA\) 通常不相等。
- 把特征分解和 SVD 混为一谈。
- 只看 shape，不写每一轴语义。
- 看到低秩就以为一定能压缩；还要看任务指标。

## 检查问题

1. \(XW\) 中 \(X\)、\(W\)、输出分别是什么 shape？
2. attention logits 的第 \(i,j\) 个元素表示什么？
3. 秩低意味着什么信息结构？
4. SVD 中奇异值衰减快说明什么？
5. 一个 head 的 attention map 若近似低秩，可能带来什么压缩机会？

## 分层练习

| 层级 | 任务 |
|---|---|
| 基础 | 用代码验证 `(T,D) @ (D,M) = (T,M)` |
| 推导 | 写出 \(QK^\top\) 在四维张量下的 shape |
| 实现 | 构造 rank-2 矩阵并验证矩阵秩 |
| 诊断 | 比较随机矩阵和低秩矩阵的奇异值曲线 |
| 迁移 | 找一篇低秩 attention 论文，标出压缩对象 |

## 阶段产出

写一页“线性代数到 Transformer 映射表”，包含 embedding、linear、Q/K/V、attention logits、低秩、SVD、投影。每个条目都写 shape 和一句直觉。

## 教材补充：embedding 表也是矩阵

词表 embedding 可以看成矩阵：

\[
E\in\mathbb{R}^{V\times D}.
\]

第 \(i\) 行 \(E_i\) 是第 \(i\) 个 token 的向量。查表操作不是矩阵乘法实现的核心，但数学上可以理解为 one-hot 向量乘 embedding 表：

\[
x_i=\operatorname{onehot}(i)^\top E.
\]

这说明 embedding lookup 也是线性代数对象。

```python
import torch

V, D = 10, 4
E = torch.randn(V, D)
token_id = torch.tensor([3])
lookup = E[token_id]
one_hot = torch.nn.functional.one_hot(token_id, num_classes=V).float()
matmul = one_hot @ E
print(torch.allclose(lookup, matmul))
```

## 手推任务：线性层参数量

若线性层把 \(D\) 维映射到 \(M\) 维：

\[
y=xW+b.
\]

参数量：

\[
D\times M+M.
\]

Transformer 中 Q/K/V/O 四个投影若都是 \(D\to D\)，参数量约为：

\[
4D^2+4D.
\]

这能帮助你读懂模型参数量来自哪里。

## Notebook 实验：奇异值衰减比较

```python
import torch

torch.manual_seed(0)
A_random = torch.randn(64, 64)
A_low = torch.randn(64, 4) @ torch.randn(4, 64)

for name, A in [('random', A_random), ('low_rank', A_low)]:
    s = torch.linalg.svdvals(A)
    ratio = s[:10] / s[0]
    print(name, ratio.round(decimals=3))
```

低秩矩阵的奇异值会快速衰减，随机矩阵通常衰减更慢。

## 论文阅读提示

读到下面词汇时，优先回到线代直觉：

| 术语 | 应关注的问题 |
|---|---|
| subspace | 哪个表示子空间被使用或压缩 |
| rank bottleneck | 信息是否被低维限制 |
| spectral norm | 变换最大放大倍数 |
| orthogonal | 方向是否互不干扰 |
| projection | 保留哪个方向的信息 |
| low-rank | 用少数方向近似原矩阵 |

## 章节小测

1. embedding lookup 如何写成 one-hot 矩阵乘法？
2. Q/K/V 投影为什么是线性变换？
3. 多头注意力为什么可以看成多个子空间上的相似度计算？
4. rank 低一定好吗？什么时候可能有害？
5. 奇异值曲线能支持什么压缩判断？

## 本章完成标准

- 能解释 embedding、linear、Q/K/V 的矩阵形式。
- 能手算线性层参数量。
- 能用 SVD 比较随机矩阵和低秩矩阵。
- 能把低秩、投影、子空间连接到 attention 分析。

