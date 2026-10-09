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
print(f"  m.is_contiguous() = {m.is_contiguous()}") # is_contiguous() 返回 True 表示数据在内存中是连续存储的

mt = m.t()
print()
print("转置之后 mt = m.t():")
print(mt)
print(f"  mt.shape     = {tuple(mt.shape)}")
print(f"  mt.stride()  = {mt.stride()}   # 变了！行方向跨 1，列方向跨 4")
print(f"  mt.is_contiguous() = {mt.is_contiguous()}")

print()
# untyped_storage()：这块 tensor 底下那串原始字节（还不管 dtype）
# data_ptr()       ：那串字节在内存里的起始地址（一个整数）
# 两个地址相等      ：说明 m 和 mt 读的是同一块内存，转置没有复制数据
print("  怎么判断两个 tensor 是否共享底层内存：")
print(f"    m.untyped_storage().data_ptr()  = {m.untyped_storage().data_ptr()}")
print(f"    mt.untyped_storage().data_ptr() = {mt.untyped_storage().data_ptr()}")
print(
    "    两个地址相等? "
    f"{m.untyped_storage().data_ptr() == mt.untyped_storage().data_ptr()}"
    "   <- True = 同一块内存"
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

print()
print("切片语法 [start:stop:step]（缺谁就用默认值）：")
print("  start 从哪开始（含），默认 0")
print("  stop  到哪结束（不含），默认到头")
print("  step  步长，默认 1")
print("  逗号 , 用来分隔不同维度，左边是第 0 维（行），右边是第 1 维（列）")
print("    m[行的切片, 列的切片]")
print("    二维就是一个逗号；三维会有两个逗号，例如 t[a, b, c]")
print("  m[:, ::2] 拆开：")
print("    第一个 :     -> 所有行")
print("    第二个 ::2   -> 列从 0 取到末尾，每隔 2 个取 1 个（第 0、2 列）")
print(f"  m[:, 0]    只要第 0 列:     {m[:, 0].tolist()}")
print(f"  m[:, 1:3]  第 1、2 列:      {m[:, 1:3].tolist()}")
print(f"  m[:, ::2]  隔列取:          {m[:, ::2].tolist()}")
print(f"  m[::2, :]  隔行取:          {m[::2, :].tolist()}")
print("  PyTorch 切片的 step 必须 > 0，不能写 ::-1（和 Python 列表不同）")
print(f"  m.flip(1)  每行列倒序:      {m.flip(1).tolist()}   # dim=1 表示沿列翻转")

ms = m[:, ::2]  # 每行只取第 0、2 列，shape 从 (3, 4) 变成 (3, 2)
print()
print("隔列取 ms = m[:, ::2]:")
print(ms)
print(f"  ms.shape     = {tuple(ms.shape)}")
print(f"  ms.stride()  = {ms.stride()}   # 行方向仍跨 4，列方向跨 2（隔一个取一个）")
print(f"  ms.is_contiguous() = {ms.is_contiguous()}   <- False，列之间有洞")
print(
    "  和 m 共享同一块内存? "
    f"{m.untyped_storage().data_ptr() == ms.untyped_storage().data_ptr()}"
    "   <- True，切片同样没有搬数据"
)
