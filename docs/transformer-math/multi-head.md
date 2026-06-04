# 多头注意力

## 学习目标

理解多头注意力不是简单“并行多个 attention”，而是把表示投影到多个子空间后分别进行信息路由，再合并输出。

## 为什么要学

多头结构的核心直觉是子空间分解。不同 head 可以关注不同关系，例如局部依赖、长程检索、语法关系或复制模式。分析 head 冗余和 head pruning 时必须有线性代数视角。

## 核心概念

- hidden dim 切分为 \(H\) 个 head。
- 每个 head 有独立 \(W_h^Q,W_h^K,W_h^V\)。
- concat 后通过 \(W^O\) 混合。
- head 之间可能冗余，也可能承担不同机制。

## 关键公式

\[
\operatorname{MHA}(X)=\operatorname{Concat}(O_1,\dots,O_H)W^O.
\]

其中：

\[
O_h=\operatorname{Attention}(XW_h^Q,XW_h^K,XW_h^V).
\]

## Transformer 对应关系

| 数学对象 | 结构含义 |
|---|---|
| 投影矩阵 | 将表示映射到 head 子空间 |
| concat | 合并多个路由结果 |
| \(W^O\) | 跨 head 混合 |
| head ablation | 判断某个子空间是否有因果贡献 |

## 最小练习

推导一个 MHA block 的参数量。设 \(d_{\text{model}}=768\)，\(H=12\)，忽略 bias，计算 Q/K/V/O 四个投影矩阵的参数量。

## 阶段产出

写一页笔记解释：为什么多头可以提高子空间覆盖，但也可能产生冗余。

