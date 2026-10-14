"""这一组小脚本共用的 GPU 同步。"""

import torch


def cuda_sync():
    """等待 GPU 上已下发的计算真正执行完。详见 08_cuda_timing.py。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
