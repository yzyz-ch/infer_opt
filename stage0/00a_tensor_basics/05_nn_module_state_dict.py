"""
05  nn.Module 与 state_dict。

运行（在 stage0 目录下）:
    python 00a_tensor_basics/05_nn_module_state_dict.py
"""

import torch
import torch.nn as nn

print("##### 5. nn.Module 与 state_dict ###")


class TinyNet(nn.Module):
    """一个最小的 nn.Module，用来看清楚 Module 到底是什么。"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

net = TinyNet()
print("一个 nn.Module 打印出来就是它的结构树：")
print(net)

print()
print("state_dict 是「参数名 -> tensor」的字典，模型权重文件存的就是它：")
# state_dict() 返回有序字典：键是层名.参数名，值是对应的权重 tensor
# .items() 每次拿出一对 (name, tensor)，例如 ("fc1.weight", 形状 (8, 4) 的矩阵)
for name, tensor in net.state_dict().items():
    print(f"  {name:12} shape={tuple(tensor.shape)}")  # :12 让名字占 12 格，方便对齐

print()
print("Qwen3 的 model.safetensors 本质上也是这样一个字典，只是有几百个条目。")
print("阶段 2 做'模型体检'时，你会去逐个核对这些名字和形状。")
