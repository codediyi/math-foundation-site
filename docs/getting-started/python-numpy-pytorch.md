# Python、NumPy 与 PyTorch 张量直觉

> 公式最终要落到数组。先把 shape、广播、矩阵乘法和自动求导弄清楚，后面学 attention 会轻很多。

## 学习目标

- 能用 Python 创建标量、向量、矩阵和三维/四维张量。
- 能解释 `shape`、`reshape`、`transpose`、`matmul`、`einsum`。
- 能理解广播规则为什么既方便又危险。
- 能用 PyTorch autograd 验证一个最小梯度公式。

## 为什么要学代码直觉

数学公式经常省略 batch 维和 head 维，但代码不会省略。比如论文写：

\[
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^\top/\sqrt{d})V
\]

实现时你必须知道：

- \(Q,K,V\) 到底是二维、三维还是四维。
- 哪两维做矩阵乘法。
- softmax 应该沿哪个维度。
- mask 如何广播。
- 输出 shape 是否还能接后面的线性层。

所以这页不追求完整 Python 教程，只建立读 Transformer 代码必需的最小张量直觉。

## 从列表到张量

```python
import torch

scalar = torch.tensor(3.0)
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

print(scalar.shape)  # torch.Size([])
print(vector.shape)  # torch.Size([3])
print(matrix.shape)  # torch.Size([2, 2])
```

读 shape 时，从左到右说清每一轴含义。不要只说“这是三维张量”，要说“第 0 维是 batch，第 1 维是 token，第 2 维是 hidden feature”。

## Transformer 常见维度

| 符号 | 代码名 | 含义 |
|---|---|---|
| \(B\) | `batch` | 一次并行处理多少样本 |
| \(T\) | `seq_len` | 序列长度，token 数 |
| \(D\) | `d_model` | 模型隐藏维度 |
| \(H\) | `num_heads` | head 数 |
| \(d_h\) | `head_dim` | 每个 head 的维度，通常 \(D/H\) |
| \(V\) | `vocab_size` | 词表大小 |

典型输入：

```python
B, T, D = 2, 5, 16
X = torch.randn(B, T, D)
print(X.shape)
```

## reshape 不是随便改含义

`reshape` 只改变数据排列的视图，不会自动理解语义。

```python
B, T, D, H = 2, 5, 16, 4
head_dim = D // H
X = torch.randn(B, T, D)
X_heads = X.reshape(B, T, H, head_dim).transpose(1, 2)
print(X_heads.shape)  # (B, H, T, head_dim)
```

这里先把最后一维拆成 `H * head_dim`，再把 head 维移动到 token 维前面。多头注意力里这一步非常常见。

## 矩阵乘法的 shape 规则

二维矩阵乘法：

\[
(m,n) @ (n,p) = (m,p)
\]

代码：

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B
print(C.shape)  # (3, 5)
```

最后一维和倒数第二维要对齐。Transformer 中：

```python
B, H, T, D = 2, 4, 5, 8
Q = torch.randn(B, H, T, D)
K = torch.randn(B, H, T, D)
logits = Q @ K.transpose(-1, -2)
print(logits.shape)  # (B, H, T, T)
```

前面的 `B,H` 被当作批量维，矩阵乘法发生在最后两维。

## einsum 读法

`einsum` 能把公式写得更接近数学：

```python
logits = torch.einsum("bhtd,bhsd->bhts", Q, K)
```

读法：

- `b` 是 batch。
- `h` 是 head。
- `t` 是 query 位置。
- `s` 是 key 位置。
- `d` 是特征维。
- 输入都含有 `d`，输出没有 `d`，说明对 `d` 求和。

这正是：

\[
S_{bhts}=\sum_d Q_{bhtd}K_{bhsd}
\]

## 广播规则

广播让不同 shape 的张量可以参与运算，但也容易悄悄出错。

```python
X = torch.randn(2, 5, 16)
bias = torch.randn(16)
Y = X + bias
print(Y.shape)  # (2, 5, 16)
```

`bias` 被自动扩展到 batch 和 token 维。这个行为适合加 bias，但如果 mask shape 写错，也可能在错误位置广播。

attention mask 常见形状：

```python
B, H, T = 2, 4, 5
logits = torch.randn(B, H, T, T)
causal = torch.tril(torch.ones(T, T)).bool()
logits = logits.masked_fill(~causal, float("-inf"))
```

这里 `(T,T)` 会广播到 `(B,H,T,T)`。

## softmax 维度

attention logits 的 shape 通常是 `(B,H,Tq,Tk)`。softmax 应该沿 key 位置做：

```python
weights = torch.softmax(logits, dim=-1)
print(weights.sum(dim=-1)[0, 0])
```

如果把 `dim` 写错，代码可能不报错，但模型含义完全变了。

## 自动求导最小例子

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)  # 2*x
```

autograd 会帮你算梯度，但学习矩阵微分时不能只依赖它。正确用法是：先手推一个小公式，再用 autograd 验证。

## attention 最小实现

```python
import torch

B, H, T, D = 2, 4, 5, 8
Q = torch.randn(B, H, T, D)
K = torch.randn(B, H, T, D)
V = torch.randn(B, H, T, D)

logits = Q @ K.transpose(-1, -2) / (D ** 0.5)
weights = torch.softmax(logits, dim=-1)
out = weights @ V

print(logits.shape)   # (2, 4, 5, 5)
print(weights.shape)  # (2, 4, 5, 5)
print(out.shape)      # (2, 4, 5, 8)
```

这是本站后续很多章节的公共代码底座。

## 常见误区

- 只打印 tensor，不打印 shape。
- 认为 `reshape` 后语义一定正确。
- 误把 `(B,T,D)` 和 `(T,B,D)` 混用。
- `softmax(dim=1)`、`softmax(dim=-1)` 随手写，没有解释维度含义。
- mask 广播到了错误轴。
- 用 `.view()` 处理非连续张量导致报错或含义不清。

## 最小练习

1. 创建一个 `(B,T,D)` 张量，并把它拆成 `(B,H,T,d_h)`。
2. 用 `matmul` 和 `einsum` 分别计算 attention logits，验证结果接近。
3. 故意把 softmax 维度改错，观察每一维求和结果。
4. 写一个 causal mask，并检查未来位置是否为 `-inf`。
5. 手推 \(y=\sum_i x_i^2\) 的梯度，再用 autograd 验证。

## 阶段产出

写一页“Transformer shape 字典”，记录从 embedding 输入到 attention 输出每个张量的 shape。后续读论文时先查这张字典。
