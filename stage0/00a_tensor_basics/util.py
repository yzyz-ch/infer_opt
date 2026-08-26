"""这一组小脚本共用的打印和 GPU 同步。"""

import torch


def banner(title: str):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def cuda_sync():
    """等待 GPU 上已下发的计算真正执行完。详见 08_cuda_timing.py。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
