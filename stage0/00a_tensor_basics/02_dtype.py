"""
02  dtype：每个数字用什么格式存。

运行（在 stage0 目录下）:
    python 00a_tensor_basics/02_dtype.py
"""

import torch

print("##### 2. dtype：每个数字用什么格式存 ###")

for dt in (torch.float32, torch.float16, torch.bfloat16, torch.int8):
    t = torch.zeros(1024, 1024, dtype=dt)
    mb = t.numel() * t.element_size() / 1024**2  # numel() 返回 tensor 中元素总数
    print(f"  {str(dt):18} 每个数 {t.element_size()} 字节(Byte), 1024*1024个数 占 {mb:6.2f} MB")

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

print("--------------------------------")

a = torch.ones(2, 2, dtype=torch.float32)
b = torch.ones(2, 2, dtype=torch.bfloat16)
#c = a @ b  这里会报错，因为 dtype 不匹配
# 需要加一步：c = a.to(torch.bfloat16) @ b.to(torch.bfloat16)
c = a.to(torch.bfloat16) @ b.to(torch.bfloat16)
c_fp32 = a.to(torch.float32) @ b.to(torch.float32)
print(c)
print(c_fp32)

print(c.dtype)
print(c_fp32.dtype)
print("--------------------------------")