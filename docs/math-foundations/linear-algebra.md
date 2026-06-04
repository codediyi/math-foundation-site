# 线性代数

## 学习目标

掌握向量空间、线性映射、秩、投影、特征值、SVD 和矩阵范数。学习重点是把“矩阵计算”提升为“表示空间变换”的视角。

## 为什么要学

Transformer 几乎每一层都由线性映射和非线性函数组合而成：

\[
Q=XW^Q,\quad K=XW^K,\quad V=XW^V,\quad Y=\operatorname{Attention}(Q,K,V)W^O.
\]

如果不知道矩阵乘法改变了什么子空间，就很难理解多头注意力、低秩压缩、KV cache 压缩和权重谱稳定性。

## 核心概念

- 向量空间、基、维数。
- 线性映射与矩阵表示。
- 秩、零空间、列空间。
- 正交投影与最小二乘。
- 特征值、谱半径、SVD、条件数。

## 关键公式

秩-零空间关系：

\[
\operatorname{rank}(A) + \operatorname{nullity}(A) = n.
\]

截断 SVD 给出最佳低秩近似：

\[
A_k = U_k \Sigma_k V_k^\top.
\]

## Transformer 对应关系

| 线代对象 | 对应问题 |
|---|---|
| 秩 | attention map 或 FFN 中的信息压缩 |
| SVD | 低秩 attention、KV cache 压缩 |
| 谱半径 | logits 尺度和训练稳定性 |
| 投影 | 多头 attention 的子空间视角 |

## 最小练习

随机生成一个矩阵，做 rank-\(k\) 截断，画出重构误差随 \(k\) 的变化。解释为什么低秩假设能节省显存但可能损失表达力。

## 关键代码块

```python
import numpy as np

A = np.random.randn(64, 64)
U, S, Vt = np.linalg.svd(A, full_matrices=False)

for k in [4, 8, 16, 32]:
    Ak = U[:, :k] @ np.diag(S[:k]) @ Vt[:k]
    err = np.linalg.norm(A - Ak, ord="fro") / np.linalg.norm(A, ord="fro")
    print(k, round(err, 4))
```

## 阶段产出

写一页笔记回答：为什么 SVD 是理解低秩 attention 和 KV 压缩的核心工具。

