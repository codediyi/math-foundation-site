# 信息论

> 信息论提供了一套描述“不确定性、分布差异、压缩和预测代价”的语言。语言模型训练的交叉熵、困惑度、KL 约束、蒸馏、采样温度和 attention 熵都可以用这套语言解释。

## 学习目标

- 理解信息量、熵、交叉熵、NLL、KL、互信息、perplexity、temperature。
- 能解释语言模型为什么最小化 cross entropy。
- 能推导 \(H(p,q)=H(p)+D_{KL}(p\|q)\)。
- 能用代码观察温度如何改变分布熵。
- 能把 KL、蒸馏、attention 熵连接到 Transformer 训练和分析。

## 从一个猜词例子开始

假设下一个 token 只有三个可能：`A/B/C`。真实分布是：

\[
p=[0.7,0.2,0.1].
\]

模型预测：

\[
q=[0.6,0.3,0.1].
\]

如果真实 token 是 `A`，模型给它的概率是 0.6。负对数似然：

\[
-\log q_A=-\log 0.6.
\]

给真实 token 的概率越高，损失越低。这就是语言模型训练目标的最小直觉。

```python
import torch

q = torch.tensor([0.6, 0.3, 0.1])
target = 0
nll = -torch.log(q[target])
print(nll.item())
```

## 信息量：越意外，信息越多

事件 \(x\) 的信息量：

\[
I(x)=-\log p(x).
\]

概率越小，信息量越大。

| 事件概率 | 信息量直觉 |
|---:|---|
| 0.9 | 很常见，信息少 |
| 0.5 | 有一定不确定性 |
| 0.01 | 很意外，信息多 |

语言模型中，如果真实 token 被模型赋予很低概率，NLL 就会很大。

## 熵：分布平均有多不确定

熵：

\[
H(p)=-\sum_x p(x)\log p(x).
\]

普通语言：按真实分布抽样时，平均需要多少信息来描述结果。

| 分布 | 熵直觉 |
|---|---|
| `[1,0,0]` | 没有不确定性，熵最低 |
| `[1/3,1/3,1/3]` | 最不确定，熵较高 |
| `[0.8,0.1,0.1]` | 有主导项，熵中等 |

```python
import torch

def entropy(p):
    return -(p * (p + 1e-12).log()).sum()

for p in [torch.tensor([1.0,0.0,0.0]), torch.tensor([1/3,1/3,1/3]), torch.tensor([0.8,0.1,0.1])]:
    print(entropy(p).item())
```

## 交叉熵：用 q 编码 p 的代价

交叉熵：

\[
H(p,q)=-\sum_x p(x)\log q(x).
\]

读法：真实分布是 \(p\)，但你用模型分布 \(q\) 去预测，平均付出的负 log 概率代价。

如果标签是 one-hot，交叉熵退化为真实类别的 NLL：

\[
H(p,q)=-\log q_y.
\]

这就是 `torch.nn.CrossEntropyLoss` 的核心。

## NLL、Cross Entropy 和语言模型

自回归语言模型建模：

\[
p(x_1,\ldots,x_T)=\prod_{t=1}^T p(x_t\mid x_{<t}).
\]

负对数似然：

\[
-\log p(x_1,\ldots,x_T)=\sum_{t=1}^T -\log p(x_t\mid x_{<t}).
\]

训练时最小化每个位置的 cross entropy，本质上就是最大化训练序列的似然。

## KL 散度：两个分布差多少

KL 散度：

\[
D_{KL}(p\|q)=\sum_x p(x)\log\frac{p(x)}{q(x)}.
\]

它衡量：如果真实是 \(p\)，却用 \(q\) 表示，会多付出多少信息代价。

注意：KL 不是对称距离。

\[
D_{KL}(p\|q)\neq D_{KL}(q\|p).
\]

forward KL 更重视覆盖 \(p\) 的高概率区域；reverse KL 更容易模式选择。

## 交叉熵分解

从定义开始：

\[
H(p,q)=-\sum_x p(x)\log q(x).
\]

加减 \(\log p(x)\)：

\[
H(p,q)=-\sum_x p(x)\log p(x)+\sum_x p(x)\log\frac{p(x)}{q(x)}.
\]

所以：

\[
H(p,q)=H(p)+D_{KL}(p\|q).
\]

当 \(p\) 固定时，最小化交叉熵等价于最小化 \(D_{KL}(p\|q)\)。

## Perplexity：交叉熵的指数形式

困惑度：

\[
\operatorname{PPL}=\exp(H).
\]

若平均 NLL 是 2.0，困惑度是 \(e^2\approx 7.39\)。直觉上，模型平均像是在约 7 个候选中困惑。

```python
import math
for ce in [1.0, 2.0, 3.0]:
    print(ce, math.exp(ce))
```

PPL 对 tokenization 很敏感，不同分词器的 PPL 不能随便直接比较。

## 温度：控制分布尖锐程度

temperature softmax：

\[
p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}.
\]

| 温度 | 效果 |
|---:|---|
| \(T<1\) | 分布更尖，熵更低 |
| \(T=1\) | 原始 softmax |
| \(T>1\) | 分布更平滑，熵更高 |

```python
import torch

logits = torch.tensor([[1.0, 2.0, 4.0]])
for T in [0.5, 1.0, 2.0]:
    p = torch.softmax(logits / T, dim=-1)
    H = -(p * (p + 1e-12).log()).sum(-1)
    print(T, p.tolist(), H.item())
```

## 蒸馏中的 KL

知识蒸馏让学生模型 \(q_s\) 学老师模型 \(q_t\)：

\[
D_{KL}(q_t\|q_s).
\]

与 one-hot 标签相比，老师分布提供了“暗知识”：非目标类别之间的相对概率。例如老师认为正确答案是 A，但 B 比 C 更相似，学生也能学到这种结构。

温度常用于软化老师分布：

\[
q_t^T=\operatorname{softmax}(z_t/T).
\]

较高温度能暴露更多类别间关系。

## attention 熵

attention 权重也是一个分布：

\[
\alpha_{ij}=\operatorname{softmax}(s_i)_j.
\]

对每个 query 位置，可以计算 attention 熵：

\[
H(\alpha_i)=-\sum_j\alpha_{ij}\log\alpha_{ij}.
\]

熵低：attention 集中在少数 key 上。熵高：attention 更分散。

```python
B, H, T = 2, 4, 5
weights = torch.softmax(torch.randn(B, H, T, T), dim=-1)
ent = -(weights * (weights + 1e-12).log()).sum(dim=-1)
print(ent.shape)  # (B,H,T)
print(ent.mean())
```

注意：attention 熵低不一定代表“更可解释”，它只说明权重更集中。

## 互信息：知道一个变量能减少多少不确定性

互信息：

\[
I(X;Y)=H(X)-H(X\mid Y).
\]

普通语言：知道 \(Y\) 后，\(X\) 的不确定性减少多少。

在表示学习中，互信息常用于描述表征与标签、上下文与目标之间的关系。但估计互信息并不容易，实践中要警惕估计偏差。

## Transformer 对应关系

| 信息论概念 | Transformer 中的位置 |
|---|---|
| 信息量 | 真实 token 的 NLL |
| 熵 | attention 集中度、采样不确定性 |
| 交叉熵 | 语言模型训练损失 |
| KL | 蒸馏、RLHF、分布约束 |
| Perplexity | 语言模型评估指标 |
| Temperature | 采样控制、蒸馏软化 |
| 互信息 | 表征分析、上下文依赖 |

## 常见误区

- 把 KL 当成对称距离。
- 认为 cross entropy 低就说明概率一定校准。
- 不同 tokenization 的 perplexity 直接比较。
- 把 attention 熵低直接解释为模型理解了因果关系。
- 温度只在推理采样中有用；蒸馏和训练分析里也常见。
- 忽略 `log(0)` 的数值问题。

## 检查问题

1. 信息量为什么是 \(-\log p(x)\)？
2. 熵高和分布均匀有什么关系？
3. one-hot 标签下 cross entropy 为什么等于 NLL？
4. 为什么最小化 cross entropy 等价于最小化 KL？
5. temperature 增大时，分布熵会怎样变化？
6. attention 熵可以说明什么，不能说明什么？

## 分层练习

| 层级 | 任务 |
|---|---|
| 基础 | 手算 `[0.5,0.5]` 和 `[0.9,0.1]` 的熵 |
| 推导 | 推导 \(H(p,q)=H(p)+D_{KL}(p\|q)\) |
| 实现 | 写函数计算 entropy、cross entropy、KL |
| 诊断 | 改变 temperature，画 entropy 曲线 |
| 迁移 | 计算一个 attention map 的平均熵并解释现象 |

## 阶段产出

整理一页“信息论到语言模型”笔记，包含 NLL、cross entropy、KL、PPL、temperature、蒸馏和 attention 熵。每个概念都要写出公式、普通语言解释和一个最小代码例子。

## 教材补充：交叉熵的 shape

语言模型 logits：

\[
Z\in\mathbb{R}^{B\times T\times V}.
\]

标签：

\[
y\in\mathbb{N}^{B\times T}.
\]

每个位置的损失：

\[
\ell_{bt}=-\log \operatorname{softmax}(Z_{bt})_{y_{bt}}.
\]

最终通常对 batch 和 token 维求均值。

```python
import torch
import torch.nn.functional as F

B, T, V = 2, 4, 10
logits = torch.randn(B, T, V)
target = torch.randint(0, V, (B, T))
loss = F.cross_entropy(logits.reshape(B*T, V), target.reshape(B*T))
print(loss)
```

## 手推任务：KL 非对称

取两个分布：

\[
p=[0.9,0.1],\quad q=[0.5,0.5].
\]

分别计算：

\[
D_{KL}(p\|q),\quad D_{KL}(q\|p).
\]

你会发现两者不同。原因是求和权重不同：forward KL 用 \(p\) 加权，reverse KL 用 \(q\) 加权。

## Notebook 实验：attention 熵与温度

```python
import torch

logits = torch.randn(2, 4, 6, 6)
for T in [0.5, 1.0, 2.0]:
    w = torch.softmax(logits / T, dim=-1)
    ent = -(w * (w + 1e-12).log()).sum(dim=-1)
    print(T, round(ent.mean().item(), 3), round(w.max(dim=-1).values.mean().item(), 3))
```

温度越低，平均最大权重越高，熵越低；温度越高，权重越分散。

## 论文阅读提示

信息论相关论文常见表述：

| 表述 | 如何理解 |
|---|---|
| minimize NLL | 最大化训练数据似然 |
| KL penalty | 约束新策略/学生分布不要偏离参考分布 |
| entropy regularization | 鼓励分布不要过早变尖 |
| low perplexity | 平均 token 预测代价低 |
| mutual information | 一个变量对另一个变量不确定性的减少 |
| calibration | 预测概率和真实频率是否匹配 |

## 章节小测

1. CrossEntropyLoss 为什么输入 logits 而不是概率？
2. PPL 为什么是 cross entropy 的指数？
3. KL penalty 在 RLHF 或蒸馏中限制了什么？
4. attention 熵低可能有哪些解释？
5. temperature 对采样多样性和错误率可能有什么影响？

## 本章完成标准

- 能写出语言模型 cross entropy 的 shape 流。
- 能手推交叉熵分解和 KL 非对称例子。
- 能计算 attention 熵并解释它的局限。
- 能把 NLL、PPL、KL、temperature 用于论文阅读。

