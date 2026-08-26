"""
02  dtype：每个数字用什么格式存。

运行（在 stage0 目录下）:
    python 00a_tensor_basics/02_dtype.py
"""

import torch

from util import banner

banner("2. dtype：每个数字用什么格式存")

for dt in (torch.float32, torch.float16, torch.bfloat16, torch.int8):
    t = torch.zeros(1000, 1000, dtype=dt)
    mb = t.numel() * t.element_size() / 1024**2
    print(f"  {str(dt):18} 每个数 {t.element_size()} 字节, 1000x1000 占 {mb:6.2f} MB")

print()
print("为什么推理几乎都用 bfloat16 而不是 float32：")
print("  - 显存占用减半，0.6B 模型从约 2.4GB 降到约 1.2GB")
print("  - 访存量减半，而大模型推理的瓶颈通常就是访存而不是算力")
print("  - bf16 的指数位和 fp32 一样宽，不像 fp16 那样容易溢出")

a = torch.ones(2, 2, dtype=torch.float32)
b = torch.ones(2, 2, dtype=torch.bfloat16)
try:
    a @ b
except RuntimeError as e:
    print()
    print(f"  float32 @ bfloat16 -> 报错: {str(e)[:70]}...")

print()
print("动手改改看：")
print("  试着让 float32 和 bfloat16 相乘成功，需要加哪一步？")
