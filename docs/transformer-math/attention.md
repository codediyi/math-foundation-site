# Attention 形状流

## 学习目标

从形状、公式、代码三层理解 scaled dot-product attention。学完后应能解释 \(QK^\top\)、mask、softmax、\(PV\) 每一步的对象和复杂度。

## 为什么要学

Attention 是 Transformer 的信息路由机制。它决定每个 token 从哪些位置读取信息，也决定训练和推理中最大的显存瓶颈之一。

## 核心概念

- Query、Key、Value 的角色。
- logits、attention probability、context vector。
- causal mask 和 padding mask。
- 时间和显存复杂度 \(O(N^2D)\)。

## 关键公式

\[
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V.
\]

其中 \(M\) 是 mask，通常把不可见位置设为极小值。

## Transformer 对应关系

| 步骤 | 形状 | 含义 |
|---|---|---|
| \(QK^\top\) | \(B,H,N,N\) | 每个 token 对所有 token 打分 |
| softmax | \(B,H,N,N\) | 信息路由权重 |
| \(PV\) | \(B,H,N,D\) | 按权重聚合 value |

## 关键代码块

```python
import torch

def attention(q, k, v, mask=None):
    d = q.shape[-1]
    logits = q @ k.transpose(-2, -1) / (d ** 0.5)
    if mask is not None:
        logits = logits.masked_fill(mask == 0, float("-inf"))
    prob = torch.softmax(logits, dim=-1)
    return prob @ v, prob
```

## 最小练习

给定 \(B=2,H=4,N=128,D=64\)，写出 logits、prob 和 output 的形状，并估算 logits 占用的元素数量。

## 阶段产出

画出一张 attention 数据流图，标出每一步的形状和复杂度。

