# 资源索引

## 中文优先资源

| 方向 | 推荐资源 | 用法 |
|---|---|---|
| 线性代数 | 清华线性代数、李思线性代数讲义 | 配合矩阵乘法、投影、SVD 章节 |
| 概率统计 | 山东大学/电子科大概率论课程 | 配合方差、协方差、CLT、MLE/MAP |
| 优化 | Boyd 凸优化课程、D2L 优化章节 | 配合 GD/Adam、smoothness、数值稳定 |
| 深度学习 | 中文《动手学深度学习》 | 配合 attention、softmax、优化实现 |
| 随机过程 | 华东师大随机过程课程 | 作为长上下文、状态更新的可选增强 |
| 实分析/泛函 | Tao Analysis、北大实变/泛函课程 | 作为研究强化段阅读 |

## 论文线索

| 研究线 | 代表主题 | 学习入口 |
|---|---|---|
| 位置编码 | RoPE、ALiBi、LongRoPE | 先学旋转、内积、相对位置 |
| 训练稳定性 | Pre-LN、DeepNorm、NormFormer、\(\mu P\) | 先学残差、归一化、Jacobian |
| 高效 Attention | FlashAttention、Linformer、Performer、GLA | 先学 log-sum-exp、低秩、核近似 |
| 机制解释 | Transformer Circuits、Induction Heads、Retrieval Heads | 先学 head ablation 和 causal intervention |
| 长上下文 | StreamingLLM、Infini-attention、LongNet | 先学位置外推、cache、检索评估 |

## 代码练习建议

| 练习 | 文件或 notebook 建议 | 验收 |
|---|---|---|
| 稳定 softmax | `softmax_stability.ipynb` | 能处理大 logits |
| 低秩近似 | `svd_low_rank.ipynb` | 误差曲线随 rank 下降 |
| attention logits | `einsum_attention.ipynb` | 形状和 matmul 版本一致 |
| LayerNorm 验证 | `layernorm_grad.ipynb` | autograd 与手推对齐 |
| 位置机制对比 | `position_encoding_toy.ipynb` | 同一 toy task 比较三种机制 |

## 复盘问题

每学完一个主题，回答下面四个问题：

1. 这个概念在 Transformer 里对应哪个真实对象？
2. 它最重要的公式是什么？
3. 它最容易造成什么实现或理解错误？
4. 我能否用一段最小代码验证它？

