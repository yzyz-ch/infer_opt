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
    gpu_t = cpu_t.to("cuda") #这里如果是多卡机器，需要指定卡号，如 gpu_t = cpu_t.to("cuda:1")
    print(gpu_t.device)
    

    # .to() 返回新 tensor，原对象不动。不接返回值 = 没搬。
    demo = torch.ones(2, 2)
    demo.to("cuda")
    print(f"  反例 demo.to('cuda') 不赋值: {demo.device}   <- 还在 CPU")
    demo = demo.to("cuda")
    print(f"  正例 demo = demo.to('cuda'): {demo.device}   <- 这才到 GPU")

    try:
        cpu_t + gpu_t
    except RuntimeError as e: # 捕获这个错误，赋给变量 e
        print(f"  cpu + cuda -> 报错: {str(e)[:70]}...") # {str(e)[:70]}... 表示截取前 70 个字符

    print()
    print("这就是为什么后面每个脚本都要写 inputs.to(model.device)：")
    print("  模型权重在 GPU 上，输入还在 CPU 上的话，forward 第一步就崩。")
else:
    print("  当前没有可用 GPU，跳过 GPU 部分")
