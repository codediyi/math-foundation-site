# 概率统计

## 学习目标

掌握随机变量、期望、方差、协方差、多变量高斯、条件期望、LLN、CLT、MLE 和 MAP。重点是理解随机初始化、mini-batch 噪声和模型输出不确定性。

## 为什么要学

Attention 中的缩放因子来自概率尺度分析。如果 \(q_i,k_i\) 独立且方差为 1，则：

\[
\operatorname{Var}(q^\top k)=d_k.
\]

因此 \(q^\top k\) 的尺度会随维度增大，除以 \(\sqrt{d_k}\) 可以保持 logits 方差相对稳定。

## 核心概念

- 期望和方差：描述均值行为和波动。
- 协方差矩阵：描述表示维度之间的相关性。
- 条件期望：理解预测目标 \(E[Y|X]\)。
- LLN/CLT：理解 batch 估计和梯度噪声。
- MLE/MAP：连接损失函数、先验和正则化。

## Transformer 对应关系

| 概率概念 | 对应问题 |
|---|---|
| 方差 | attention logits 缩放 |
| 协方差 | 表征相关性、梯度噪声结构 |
| 条件期望 | 预测目标与泛化误差 |
| MAP | weight decay 和先验直觉 |

## 关键代码块

```python
import torch

for d in [16, 64, 256, 1024]:
    q = torch.randn(10000, d)
    k = torch.randn(10000, d)
    dot = (q * k).sum(dim=-1)
    scaled = dot / (d ** 0.5)
    print(d, dot.var().item(), scaled.var().item())
```

## 最小练习

推导 \(q^\top k\) 的方差，并用上面的模拟验证。然后写一句话解释为什么这个结论直接影响 softmax 的饱和程度。

## 阶段产出

写一页笔记：batch size 如何影响梯度方差，以及为什么大 batch 不只是“更快”。

