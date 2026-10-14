"""
01  shape：tensor 每个维度有多大。

运行（在 stage0 目录下）:
    python 00a_tensor_basics/01_shape.py
"""

import torch

print("##### 1. shape：tensor 每个维度有多大 ###")

x = torch.zeros(2, 3, 4)
print("x = torch.zeros(2, 3, 4)")
print(f"  x.shape  = {x.shape}        # 三个维度，分别是 2 / 3 / 4")
print(f"  x.ndim   = {x.ndim}                        # 几个维度")
print(f"  x.numel() = {x.numel()}                     # 一共多少个数 = 2*3*4")

print()
print("推理中会反复见到的 shape 含义：")
print("  input_ids : [batch, seq_len]              一批有几条输入，每条多少个 token")
print("  hidden    : [batch, seq_len, hidden_size] 每个 token 的向量表示")
print("  logits    : [batch, seq_len, vocab_size]  每个位置对词表里每个词的打分")

y = x.view(2, 12)
print()
print(f"x.view(2, 12).shape = {y.shape}   # 元素总数不变，只是换个看待方式")
