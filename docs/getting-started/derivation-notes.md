# 推导与笔记方法

## 学习目标

本页给出一种面向论文阅读的推导模板。目标不是写严密教材证明，而是把论文中的关键公式拆成“对象、形状、变换、假设、结论”五部分。

## 推导模板

| 区块 | 需要写什么 |
|---|---|
| 对象 | 变量是什么，属于哪个空间，形状是多少 |
| 变换 | 从输入到输出经过哪些线性/非线性变换 |
| 假设 | 独立性、归一化、尺度、近似或边界条件 |
| 结论 | 公式说明了什么性质 |
| 误区 | 哪一步最容易误解 |

## 关键公式入口

以 scaled dot-product attention 为例：

\[
S = \frac{QK^\top}{\sqrt{d_k}}, \qquad
P = \operatorname{softmax}(S), \qquad
O = PV.
\]

推导时先不要急着求梯度，先检查形状：

| 张量 | 形状 | 含义 |
|---|---|---|
| \(Q\) | \(B \times H \times N \times D\) | 每个 token 的 query |
| \(K\) | \(B \times H \times N \times D\) | 每个 token 的 key |
| \(S\) | \(B \times H \times N \times N\) | token 到 token 的打分 |
| \(P\) | \(B \times H \times N \times N\) | 信息路由权重 |
| \(O\) | \(B \times H \times N \times D\) | 聚合后的表示 |

## 最小练习

手推 \(Y=XW\) 的梯度。要求写出：

- \(X\)、\(W\)、\(Y\)、上游梯度 \(G_Y\) 的形状。
- 为什么 \(G_X = G_Y W^\top\)。
- 为什么 \(G_W = X^\top G_Y\)。

## 阶段产出

建立一个“公式卡片库”，每张卡片只解决一个公式。卡片标题建议使用“对象 + 问题”，例如“Attention logits 为什么除以 \(\sqrt{d_k}\)”。

