# Attention 反向传播

## 学习目标

理解 attention backward 的主要梯度路径：从输出到 \(V\)，从 attention probability 到 logits，再回到 \(Q\) 和 \(K\)。

## 为什么要学

很多结构改进会改变 attention logits、softmax 或 value 聚合。只有理解反传，才能判断改动会如何影响梯度尺度和训练稳定性。

## 核心概念

- 上游梯度 \(G_O\)。
- \(O=PV\) 的矩阵乘法反传。
- softmax Jacobian。
- \(S=QK^\top/\sqrt{d_k}\) 的反传。

## 关键公式

若 \(O=PV\)，则：

\[
G_P = G_O V^\top,\qquad G_V = P^\top G_O.
\]

若 \(S=QK^\top/\sqrt{d_k}\)，则：

\[
G_Q = G_S K / \sqrt{d_k},\qquad G_K = G_S^\top Q / \sqrt{d_k}.
\]

## Transformer 对应关系

| 梯度路径 | 对应风险 |
|---|---|
| \(G_V\) | value 表示是否被有效训练 |
| softmax backward | 饱和后梯度变小 |
| \(G_Q,G_K\) | logits 尺度影响投影学习 |
| mask | 被 mask 位置无梯度 |

## 最小练习

用 PyTorch autograd 验证 \(O=PV\) 的梯度公式。再把 softmax 加进去，观察 logits 很大时梯度是否变小。

## 阶段产出

整理一张 attention backward 流程图，标出每个梯度张量的形状。

