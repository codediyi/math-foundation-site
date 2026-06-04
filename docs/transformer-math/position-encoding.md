# 位置编码

## 学习目标

理解绝对位置、相对位置、RoPE 和 ALiBi 的数学差异。重点是把位置机制看作“给 attention 加入位置依赖”的不同方式。

## 为什么要学

没有位置机制，self-attention 对 token 顺序不敏感。位置编码决定模型能否表达顺序、距离、相对位移和长度外推。

## 核心概念

- Sinusoidal position embedding。
- Learned absolute position embedding。
- RoPE 的二维旋转。
- ALiBi 的距离偏置。
- 长度外推和频谱视角。

## 关键公式

RoPE 可理解为对 \(q,k\) 做位置相关旋转：

\[
\langle R_m q, R_n k\rangle = q^\top R_{n-m}k.
\]

这说明 attention 分数可以显式依赖相对位置 \(n-m\)。

## Transformer 对应关系

| 方法 | 注入位置的方式 | 常见研究问题 |
|---|---|---|
| Sinusoidal | 加到 embedding | 固定频率能否外推 |
| RoPE | 旋转 Q/K | 相对位置与长度外推 |
| ALiBi | 加 attention bias | 距离惩罚与外推 |
| LongRoPE | 调整频率尺度 | 超长上下文 |

## 关键代码块

```python
import torch

def apply_rope_pair(x, theta):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)
```

## 最小练习

实现 sinusoidal、RoPE、ALiBi 三种位置机制的 toy 版本，比较它们对序列长度翻倍时 attention bias 的变化。

## 阶段产出

完成一张表：位置机制、数学形式、外推直觉、优点、风险。

