# 长上下文

## 学习目标

理解长上下文研究的三条主线：位置外推、复杂度降低、缓存与记忆。目标是能把长上下文论文分到正确技术族，并设计小规模复现实验。

## 为什么要学

长上下文不是简单把 context window 调大。位置编码、attention 复杂度、KV cache、检索能力和评估 benchmark 都会成为瓶颈。

## 核心概念

- RoPE/ALiBi/LongRoPE 位置外推。
- 稀疏、低秩、线性 attention。
- KV cache 压缩和淘汰。
- StreamingLLM、Infini-attention、LongNet。
- Needle-in-a-haystack 和检索评估。

## Transformer 对应关系

| 问题 | 常见方法 |
|---|---|
| 位置外推 | 频率重标定、距离 bias |
| 显存爆炸 | FlashAttention、稀疏/线性 attention |
| cache 过大 | KV 压缩、retrieval-head aware pruning |
| 检索失败 | head 机制分析、数据构造 |

## 最小练习

设计一个 toy 长上下文检索任务：在长序列中插入一个 key-value pair，让模型在末尾回答。记录上下文长度增长时准确率如何变化。

## 阶段产出

完成一张长上下文方法地图，按“位置、复杂度、cache、评估”四类组织。

