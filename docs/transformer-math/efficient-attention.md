# 高效 Attention

## 学习目标

理解高效 attention 的四类主要思想：低秩、稀疏、核近似、I/O-aware 精确计算。学完后应能判断一个方法是在改数学近似还是在改系统实现。

## 为什么要学

标准 attention 的 logits 是 \(N\times N\)。当上下文从 1k 增长到 10k，attention 矩阵规模增长 100 倍。长上下文研究必须同时考虑数学近似、显存访问和实际 wall-clock。

## 核心概念

- 低秩近似：Linformer 类方法。
- 稀疏注意力：Longformer、BigBird。
- 核近似和线性 attention：Performer、GLA。
- I/O-aware 精确 attention：FlashAttention。

## 关键公式

线性 attention 的典型形式：

\[
\operatorname{softmax}(QK^\top)V \approx \phi(Q)(\phi(K)^\top V).
\]

它把 \(N\times N\) 的显式 attention map 避开，但近似质量取决于特征映射 \(\phi\)。

## Transformer 对应关系

| 方法族 | 改了什么 | 风险 |
|---|---|---|
| 低秩 | 压缩序列维 | 表达力不足 |
| 稀疏 | 限制可见位置 | 丢失全局信息 |
| 核近似 | 替代 softmax kernel | 近似误差和稳定性 |
| FlashAttention | 改内存访问 | 数学精确但依赖 kernel |

## 最小练习

比较普通 attention 和分块 online softmax 的内存占用路径。只需写出中间张量规模，不必实现完整 CUDA kernel。

## 阶段产出

做一张方法分类表：方法、数学假设、复杂度、适用场景、最小复现实验。

