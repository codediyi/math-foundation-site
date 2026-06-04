# 矩阵微分

## 学习目标

掌握 gradient、Jacobian、Hessian、链式法则和常见矩阵乘法梯度。目标是能手推 attention 和 LayerNorm 的关键反传路径。

## 为什么要学

只会 PyTorch autograd 不等于理解训练。读训练稳定性、归一化、attention 变体论文时，经常需要判断某个结构如何改变梯度传播。

## 核心概念

- 标量对向量的梯度。
- 向量对向量的 Jacobian。
- 标量对向量的 Hessian。
- 矩阵乘法的反向传播。
- 残差连接中的 \(I+J_f\)。

## 关键公式

若 \(Y=XW\)，上游梯度为 \(G_Y=\partial L/\partial Y\)，则：

\[
G_X = G_Y W^\top,\qquad G_W = X^\top G_Y.
\]

## Transformer 对应关系

| 微分对象 | 对应模块 |
|---|---|
| 矩阵乘法梯度 | Q/K/V 投影和 FFN |
| softmax Jacobian | attention 权重反传 |
| 残差 Jacobian | 深层梯度传播 |
| LayerNorm 梯度 | 训练稳定性分析 |

## 关键代码块

```python
import torch

X = torch.randn(3, 4, requires_grad=True)
W = torch.randn(4, 5, requires_grad=True)
Y = X @ W
loss = (Y ** 2).sum()
loss.backward()

with torch.no_grad():
    GY = 2 * Y
    print(torch.allclose(X.grad, GY @ W.T))
    print(torch.allclose(W.grad, X.T @ GY))
```

## 最小练习

手推 \(Y=XW\) 的梯度，并用 autograd 验证。之后把 \(XW\) 换成 \(QK^\top\)，写出 \(G_Q\) 和 \(G_K\)。

## 阶段产出

整理一张“Transformer 常见梯度公式卡”，至少包含线性层、softmax、LayerNorm 和残差。

