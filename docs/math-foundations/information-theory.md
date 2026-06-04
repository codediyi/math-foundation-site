# 信息论

## 学习目标

理解熵、交叉熵、KL 散度、互信息和温度。重点是把训练目标、蒸馏、概率校准和信息压缩放进同一套语言里。

## 为什么要学

分类模型通常最小化交叉熵：

\[
H(p,q) = H(p) + D_{\mathrm{KL}}(p\|q).
\]

当真实分布 \(p\) 固定时，最小化交叉熵等价于最小化 KL 散度。这解释了为什么最大似然和交叉熵训练是同一件事的两种表述。

## 核心概念

- 熵：分布的不确定性。
- 交叉熵：用 \(q\) 编码来自 \(p\) 的样本所需代价。
- KL 散度：两个分布的非对称差异。
- 温度：控制 softmax 分布尖锐程度。
- 互信息：变量之间共享信息的度量。

## Transformer 对应关系

| 信息论对象 | 对应问题 |
|---|---|
| 交叉熵 | 语言模型训练损失 |
| KL | 蒸馏、RLHF、分布对齐 |
| 熵 | attention 集中程度、输出不确定性 |
| 温度 | logits 缩放、采样、校准 |

## 关键代码块

```python
import torch

def entropy(p):
    return -(p * (p + 1e-12).log()).sum(dim=-1)

logits = torch.tensor([[1.0, 2.0, 4.0]])
for temp in [0.5, 1.0, 2.0]:
    p = torch.softmax(logits / temp, dim=-1)
    print(temp, p, entropy(p))
```

## 最小练习

比较 temperature 为 0.5、1.0、2.0 时的 softmax 分布和熵。解释为什么温度降低会让模型更“自信”但不一定更准确。

## 阶段产出

写一页笔记回答：KL 为什么不是距离，以及 forward KL 与 reverse KL 的行为差异。

