"""
07  inference_mode()：不记录反向传播需要的信息。

运行（在 stage0 目录下）:
    python 00a_tensor_basics/07_inference_mode.py
"""

import torch

print("##### 7. inference_mode()：不记录反向传播需要的信息 ###")

w = torch.ones(2, 2, requires_grad=True)

out_normal = w * 2
print(
    f"  普通模式下  : requires_grad={out_normal.requires_grad}, "
    f"grad_fn={out_normal.grad_fn}"
)
print("    -> PyTorch 记住了'这个结果是怎么算出来的'，为反向传播做准备")

with torch.inference_mode():
    out_infer = w * 2
print(
    f"  inference_mode: requires_grad={out_infer.requires_grad}, "
    f"grad_fn={out_infer.grad_fn}"
)
print("    -> 什么都不记，省下显存和一部分开销")

print()
print("推理时这些记录纯属浪费：我们不会更新权重，也就不需要反向传播。")
print("no_grad() 也能达到类似效果，inference_mode() 更彻底、更快一点。")
