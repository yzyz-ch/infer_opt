"""
04  stride 与 contiguous（反直觉实验一）。

转置之后数据其实没有搬动。阶段 7 接 kernel 时只检查 shape 会出事。

运行（在 stage0 目录下）:
    python 00a_tensor_basics/04_stride_contiguous.py
"""

import torch

from util import banner

banner("4. stride 与 contiguous（反直觉实验一）")

m = torch.arange(12).reshape(3, 4)
print("原始 tensor m = arange(12).reshape(3, 4):")
print(m)
print(f"  m.shape      = {tuple(m.shape)}")
print(f"  m.stride()   = {m.stride()}   # 行方向走一步跨 4 个元素，列方向跨 1 个")
print(f"  m.is_contiguous() = {m.is_contiguous()}")

mt = m.t()
print()
print("转置之后 mt = m.t():")
print(mt)
print(f"  mt.shape     = {tuple(mt.shape)}")
print(f"  mt.stride()  = {mt.stride()}   # 变了！行方向跨 1，列方向跨 4")
print(f"  mt.is_contiguous() = {mt.is_contiguous()}")

print()
print(
    "  m 和 mt 共享同一块底层内存吗: "
    f"{m.untyped_storage().data_ptr() == mt.untyped_storage().data_ptr()}"
)
print("  转置只是改了'怎么读这块内存'的说明书（shape + stride），没有真的搬数据。")

mtc = mt.contiguous()
print()
print(f"  mt.contiguous() 之后 is_contiguous = {mtc.is_contiguous()}")
print(
    "  这次底层内存换了吗: "
    f"{mt.untyped_storage().data_ptr() != mtc.untyped_storage().data_ptr()}"
    "   <- True 说明真的复制了一份"
)

print()
print("这一节为什么重要（记到阶段 7）：")
print("  你自己写的 CUDA kernel 通常假设数据是连续排放的。")
print("  如果只检查 shape 就放行，遇到一个转置过来的 tensor，")
print("  shape 看着完全正确，读出来的数却全是错位的，而且不报错。")

print()
print("动手改改看：")
print("  把 m.t() 换成 m[:, ::2]（隔列取），看看 stride 变成什么，")
print("  is_contiguous 还是不是 True。")
