"""
08  CUDA 计时与 synchronize（反直觉实验二）。

不同步就计时会得到假数字。后面所有性能测量都必须先 synchronize。

运行（在 stage0 目录下）:
    python 00a_tensor_basics/08_cuda_timing.py
"""

import time

import torch

from util import banner, cuda_sync

banner("8. CUDA 计时与 synchronize（反直觉实验二）")

if not torch.cuda.is_available():
    print("  当前没有 GPU，跳过这一节")
else:
    big = torch.randn(4096, 4096, device="cuda")

    for _ in range(3):
        big @ big
    cuda_sync()

    t0 = time.perf_counter()
    for _ in range(10):
        big @ big
    wrong = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(10):
        big @ big
    cuda_sync()
    right = time.perf_counter() - t0

    print(f"  不调用 synchronize 测得: {wrong*1000:8.3f} ms")
    print(f"  调用   synchronize 测得: {right*1000:8.3f} ms")
    print(f"  相差 {right / max(wrong, 1e-9):.1f} 倍")
    print()
    print("  原因：CUDA 是异步的。Python 里 `big @ big` 返回时，")
    print("  只表示'任务已经排进 GPU 队列'，不表示'GPU 算完了'。")
    print("  不同步就停表，量的是下发任务的时间，不是计算时间。")
    print()
    print("  后面所有性能测量都必须先 synchronize。阶段 5 定位性能问题时，")
    print("  忘了这一步会得出完全错误的结论。")

    print()
    print("动手改改看：")
    print("  把矩阵从 4096 改成 256，再看两种计时的差距。")
    print("  矩阵变小之后差距是变大还是变小？为什么？")
