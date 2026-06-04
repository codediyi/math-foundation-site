# 六个月路线

## 总目标

第 26 周完成一个从零 attention 小项目或小型 Transformer 改进原型。默认每周投入 10 小时：阅读 3 小时、推导 2 小时、实现 3 小时、复盘 2 小时。

## 阶段安排

| 周数 | 主题 | 核心能力 | 阶段产出 |
|---|---|---|---|
| 第 0 周 | 准备 | 建立笔记、代码、实验目录 | 学习日志和环境 |
| 第 1-6 周 | 线性代数与张量 | 矩阵乘法、秩、投影、SVD、einsum | 低秩近似实验 |
| 第 7-12 周 | 概率统计与信息论 | 方差、协方差、KL、交叉熵、稳定 softmax | softmax/CE 实现 |
| 第 13-17 周 | 分析、矩阵微分、优化 | Taylor、Jacobian、Hessian、GD/Adam、LayerNorm | attention/LN 反传推导 |
| 第 18-26 周 | Transformer 数学 | attention、MHA、位置编码、归一化、高效 attention | mini Transformer block |

## 每周固定交付

- 一页概念笔记。
- 一页公式推导。
- 一段可运行代码。
- 一个小实验或数值验证。

## 里程碑

| 里程碑 | 完成标准 |
|---|---|
| M1 线代可用 | 能解释 SVD 与低秩 attention 的关系 |
| M2 概率可用 | 能推导 \(q^\top k\) 方差和 \(\sqrt{d_k}\) 缩放 |
| M3 数值稳定可用 | 能实现稳定 softmax / log-sum-exp |
| M4 反传可用 | 能手推 \(Q,K,V\) 和 LayerNorm 梯度 |
| M5 Transformer 可用 | 能从零写 attention、MHA、FFN、LN |
| M6 论文可用 | 能读 RoPE、ALiBi、FlashAttention、Pre-LN 核心公式 |

## 最终交付物

1. 一个可运行代码仓库：attention、LayerNorm、position encoding、toy training loop。
2. 一份 6 页左右技术报告。
3. 一张实验表：至少比较两个 baseline，包含 loss、梯度范数、注意力熵或长度外推指标。

