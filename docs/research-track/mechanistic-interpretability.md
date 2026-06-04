# 机制解释

## 学习目标

理解机制解释中的基本问题：某个 head 或 circuit 是否真的承担了某种算法功能。重点学习 causal intervention，而不是只看 attention heatmap。

## 为什么要学

Transformer Circuits、Induction Heads、Retrieval Heads 等工作说明，模型内部可能存在可解释的算法结构。要把这种发现转成研究结论，必须做可验证干预。

## 核心概念

- Attention head pattern。
- Head ablation。
- Activation patching。
- Induction head。
- Retrieval head。
- 因果证据与相关性证据。

## Transformer 对应关系

| 机制 | 典型现象 |
|---|---|
| Induction head | 根据前文模式复制/补全 |
| Retrieval head | 长上下文中负责检索关键信息 |
| Head ablation | 移除某头后性能显著下降 |
| Activation patching | 替换激活验证因果路径 |

## 最小练习

在一个小模型或公开分析 notebook 中，选择一个 attention head，记录 attention pattern，然后 ablate 它并比较任务指标变化。

## 阶段产出

写一页实验记录：这个 head 的作用是相关性观察，还是有干预证据支持。

