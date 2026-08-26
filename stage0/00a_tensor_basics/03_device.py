"""
03  device：数据放在哪块硬件上。

运行（在 stage0 目录下）:
    python 00a_tensor_basics/03_device.py
"""

import torch

from util import banner

banner("3. device：数据放在哪块硬件上")

cpu_t = torch.ones(2, 2)
print(f"  默认创建在 CPU: {cpu_t.device}")

if torch.cuda.is_available():
    gpu_t = cpu_t.to("cuda")
    print(f"  .to('cuda') 之后: {gpu_t.device}")
    print(f"  原来那个还在   : {cpu_t.device}   <- .to() 不是原地操作，容易忘")

    try:
        cpu_t + gpu_t
    except RuntimeError as e:
        print(f"  cpu + cuda -> 报错: {str(e)[:70]}...")

    print()
    print("这就是为什么后面每个脚本都要写 inputs.to(model.device)：")
    print("  模型权重在 GPU 上，输入还在 CPU 上的话，forward 第一步就崩。")
else:
    print("  当前没有可用 GPU，跳过 GPU 部分")
