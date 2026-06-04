# 优化与数值稳定

## 学习目标

掌握梯度下降、SGD、Adam、动量、L-smooth、强凸、条件数、梯度裁剪和 log-sum-exp。重点是理解训练不稳定的数学来源，而不是只会换优化器。

## 为什么要学

Transformer 训练稳定性来自多个尺度控制：初始化、残差、归一化、学习率、warmup、梯度裁剪和数值实现。如果 logits 太大，softmax 会饱和；如果更新尺度太大，残差块会放大扰动。

## 核心概念

- 梯度下降：沿局部下降方向更新。
- L-smooth：控制梯度变化速度。
- 条件数：影响不同方向收敛速度。
- Adam：动量和自适应预条件。
- log-sum-exp：稳定计算 softmax 和交叉熵。

## 关键公式

稳定 log-sum-exp：

\[
\log\sum_i e^{z_i} = m + \log\sum_i e^{z_i-m}, \quad m=\max_i z_i.
\]

## Transformer 对应关系

| 优化概念 | 对应问题 |
|---|---|
| warmup | 训练早期更新尺度控制 |
| 梯度裁剪 | 避免异常 batch 放大参数更新 |
| log-sum-exp | 稳定 softmax / cross entropy |
| 条件数 | 权重谱、激活尺度、收敛速度 |

## 关键代码块

```python
import torch

def stable_softmax(x, dim=-1):
    z = x - x.max(dim=dim, keepdim=True).values
    return z.exp() / z.exp().sum(dim=dim, keepdim=True)

x = torch.tensor([[1000.0, 1001.0, 1002.0]])
print(stable_softmax(x))
```

## 最小练习

实现一个不稳定 softmax 和稳定 softmax，输入 `[1000, 1001, 1002]`，比较结果。解释 overflow 为什么不是“小实现细节”，而是训练正确性的前提。

## 阶段产出

做一个二次函数实验，比较不同条件数下 GD 的收敛曲线，写出学习率与发散的关系。

