# 张量运算

## 学习目标

理解张量不是“高维数组”的同义词，而是带有轴语义的多线性对象。重点掌握 batch、head、sequence、hidden 四类轴如何在 Transformer 中组合。

## 为什么要学

Transformer 的很多 bug 不是公式错，而是形状错。比如 attention logits 通常写成：

\[
S_{b,h,i,j} = \sum_d Q_{b,h,i,d}K_{b,h,j,d}.
\]

这背后是一次张量收缩，不只是普通矩阵乘法。

## 核心概念

- 轴语义：batch、sequence、head、feature。
- broadcast、reshape、transpose。
- Hadamard product、Kronecker product、trace。
- Einstein summation 和张量收缩。

## Transformer 对应关系

| 张量操作 | 对应模块 |
|---|---|
| reshape/split | 将 hidden dim 切成多个 head |
| transpose | 调整 matmul 的收缩轴 |
| einsum | 清晰表达 attention logits |
| broadcast | mask、bias、position bias |

## 关键代码块

```python
import torch

B, H, N, D = 2, 4, 8, 16
q = torch.randn(B, H, N, D)
k = torch.randn(B, H, N, D)

logits = torch.einsum("bhid,bhjd->bhij", q, k) / (D ** 0.5)
print(logits.shape)  # [2, 4, 8, 8]
```

## 最小练习

不用 `einsum`，只用 `transpose` 和 `matmul` 重写上面的 attention logits，并确认结果相同。

## 阶段产出

做一张 Transformer 张量形状流图，从输入 token id 到 embedding、QKV、attention logits、attention output、FFN output。

