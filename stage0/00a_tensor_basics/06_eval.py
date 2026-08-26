"""
06  eval()：切换到推理模式。

运行（在 stage0 目录下）:
    python 00a_tensor_basics/06_eval.py
"""

import torch
import torch.nn as nn

from util import banner

banner("6. eval()：切换到推理模式")

drop_net = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.5))
inp = torch.ones(1, 4)

drop_net.train()
outs = [drop_net(inp) for _ in range(3)]
print("train() 模式下，同样的输入跑三次：")
for i, o in enumerate(outs):
    print(f"  第{i+1}次: {o.detach().numpy().round(4)}")
print("  -> 每次都不一样，因为 Dropout 在随机丢弃一半神经元")

drop_net.eval()
outs = [drop_net(inp) for _ in range(3)]
print()
print("eval() 模式下，同样的输入跑三次：")
for i, o in enumerate(outs):
    print(f"  第{i+1}次: {o.detach().numpy().round(4)}")
print("  -> 完全一致。推理要的就是这种确定性。")
