"""
00a：推理需要的 PyTorch 地基。

这个脚本不加载大模型，纯粹用小 tensor 把下面这些概念跑一遍：
    shape / dtype / device / stride / contiguous
    nn.Module / state_dict / eval() / inference_mode()
    CPU 与 GPU 之间搬运
    CUDA 同步与计时

其中有两个"反直觉实验"是重点，别跳过：
    第 4 节：转置之后数据其实没有搬动
    第 8 节：不同步就计时会得到假数字

运行:
    ../local/venvs/.venv-hf/bin/python 00a_tensor_basics.py
"""

import time

import torch
import torch.nn as nn

from common import DEVICE, banner, cuda_sync


# ---------------------------------------------------------------------------
banner("1. shape：tensor 每个维度有多大")
# ---------------------------------------------------------------------------

x = torch.zeros(2, 3, 4)
print(f"x = torch.zeros(2, 3, 4)")
print(f"  x.shape  = {x.shape}        # 三个维度，分别是 2 / 3 / 4")
print(f"  x.ndim   = {x.ndim}                        # 几个维度")
print(f"  x.numel() = {x.numel()}                     # 一共多少个数 = 2*3*4")

# 推理里最常见的三种 shape，先混个眼熟：
print()
print("推理中会反复见到的 shape 含义：")
print("  input_ids : [batch, seq_len]              一批有几条输入，每条多少个 token")
print("  hidden    : [batch, seq_len, hidden_size] 每个 token 的向量表示")
print("  logits    : [batch, seq_len, vocab_size]  每个位置对词表里每个词的打分")

# 改 shape 但不改数据：view / reshape
y = x.view(2, 12)
print()
print(f"x.view(2, 12).shape = {y.shape}   # 元素总数不变，只是换个看待方式")


# ---------------------------------------------------------------------------
banner("2. dtype：每个数字用什么格式存")
# ---------------------------------------------------------------------------

for dt in (torch.float32, torch.float16, torch.bfloat16, torch.int8):
    t = torch.zeros(1000, 1000, dtype=dt)
    mb = t.numel() * t.element_size() / 1024**2
    print(f"  {str(dt):18} 每个数 {t.element_size()} 字节, 1000x1000 占 {mb:6.2f} MB")

print()
print("为什么推理几乎都用 bfloat16 而不是 float32：")
print("  - 显存占用减半，0.6B 模型从约 2.4GB 降到约 1.2GB")
print("  - 访存量减半，而大模型推理的瓶颈通常就是访存而不是算力")
print("  - bf16 的指数位和 fp32 一样宽，不像 fp16 那样容易溢出")

# dtype 不匹配会直接报错，这是新手最常撞的墙之一
a = torch.ones(2, 2, dtype=torch.float32)
b = torch.ones(2, 2, dtype=torch.bfloat16)
try:
    a @ b
except RuntimeError as e:
    print()
    print(f"  float32 @ bfloat16 -> 报错: {str(e)[:70]}...")


# ---------------------------------------------------------------------------
banner("3. device：数据放在哪块硬件上")
# ---------------------------------------------------------------------------

cpu_t = torch.ones(2, 2)
print(f"  默认创建在 CPU: {cpu_t.device}")

if torch.cuda.is_available():
    gpu_t = cpu_t.to("cuda")
    print(f"  .to('cuda') 之后: {gpu_t.device}")
    # 注意：.to() 返回新 tensor，不是原地修改
    print(f"  原来那个还在   : {cpu_t.device}   <- .to() 不是原地操作，容易忘")

    # 跨设备运算同样直接报错
    try:
        cpu_t + gpu_t
    except RuntimeError as e:
        print(f"  cpu + cuda -> 报错: {str(e)[:70]}...")

    print()
    print("这就是为什么后面每个脚本都要写 inputs.to(model.device)：")
    print("  模型权重在 GPU 上，输入还在 CPU 上的话，forward 第一步就崩。")
else:
    print("  当前没有可用 GPU，跳过 GPU 部分")


# ---------------------------------------------------------------------------
banner("4. stride 与 contiguous（反直觉实验一）")
# ---------------------------------------------------------------------------

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

# 关键证据：底层存储是同一块内存，一个字节都没搬
print()
print(f"  m 和 mt 共享同一块底层内存吗: "
      f"{m.untyped_storage().data_ptr() == mt.untyped_storage().data_ptr()}")
print("  转置只是改了'怎么读这块内存'的说明书（shape + stride），没有真的搬数据。")

mtc = mt.contiguous()
print()
print(f"  mt.contiguous() 之后 is_contiguous = {mtc.is_contiguous()}")
print(f"  这次底层内存换了吗: "
      f"{mt.untyped_storage().data_ptr() != mtc.untyped_storage().data_ptr()}"
      "   <- True 说明真的复制了一份")

print()
print("这一节为什么重要（记到阶段 7）：")
print("  你自己写的 CUDA kernel 通常假设数据是连续排放的。")
print("  如果只检查 shape 就放行，遇到一个转置过来的 tensor，")
print("  shape 看着完全正确，读出来的数却全是错位的，而且不报错。")


# ---------------------------------------------------------------------------
banner("5. nn.Module 与 state_dict")
# ---------------------------------------------------------------------------

class TinyNet(nn.Module):
    """一个最小的 nn.Module，用来看清楚 Module 到底是什么。"""

    def __init__(self):
        super().__init__()
        # 在 __init__ 里定义"这个网络由哪些层组成"
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        # 在 forward 里定义"数据怎么流过这些层"
        return self.fc2(torch.relu(self.fc1(x)))


net = TinyNet()
print("一个 nn.Module 打印出来就是它的结构树：")
print(net)

print()
print("state_dict 是「参数名 -> tensor」的字典，模型权重文件存的就是它：")
for name, tensor in net.state_dict().items():
    print(f"  {name:12} shape={tuple(tensor.shape)}")

print()
print("Qwen3 的 model.safetensors 本质上也是这样一个字典，只是有几百个条目。")
print("阶段 2 做'模型体检'时，你会去逐个核对这些名字和形状。")


# ---------------------------------------------------------------------------
banner("6. eval()：切换到推理模式")
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
banner("7. inference_mode()：不记录反向传播需要的信息")
# ---------------------------------------------------------------------------

w = torch.ones(2, 2, requires_grad=True)

out_normal = w * 2
print(f"  普通模式下  : requires_grad={out_normal.requires_grad}, "
      f"grad_fn={out_normal.grad_fn}")
print("    -> PyTorch 记住了'这个结果是怎么算出来的'，为反向传播做准备")

with torch.inference_mode():
    out_infer = w * 2
print(f"  inference_mode: requires_grad={out_infer.requires_grad}, "
      f"grad_fn={out_infer.grad_fn}")
print("    -> 什么都不记，省下显存和一部分开销")

print()
print("推理时这些记录纯属浪费：我们不会更新权重，也就不需要反向传播。")
print("no_grad() 也能达到类似效果，inference_mode() 更彻底、更快一点。")


# ---------------------------------------------------------------------------
banner("8. CUDA 计时与 synchronize（反直觉实验二）")
# ---------------------------------------------------------------------------

if not torch.cuda.is_available():
    print("  当前没有 GPU，跳过这一节")
else:
    big = torch.randn(4096, 4096, device="cuda")

    # 先跑几次预热。第一次调用要初始化 CUDA 上下文、挑选 kernel，
    # 不预热的话第一次的耗时会离谱地大。
    for _ in range(3):
        big @ big
    cuda_sync()

    # 错误写法：不同步
    t0 = time.perf_counter()
    for _ in range(10):
        big @ big
    wrong = time.perf_counter() - t0

    # 正确写法：算完再停表
    t0 = time.perf_counter()
    for _ in range(10):
        big @ big
    cuda_sync()
    right = time.perf_counter() - t0

    print(f"  不调用 synchronize 测得: {wrong*1000:8.3f} ms")
    print(f"  调用   synchronize 测得: {right*1000:8.3f} ms")
    print(f"  相差 {right/max(wrong, 1e-9):.1f} 倍")
    print()
    print("  原因：CUDA 是异步的。Python 里 `big @ big` 返回时，")
    print("  只表示'任务已经排进 GPU 队列'，不表示'GPU 算完了'。")
    print("  不同步就停表，量的是下发任务的时间，不是计算时间。")
    print()
    print("  后面所有性能测量都必须先 synchronize。阶段 5 定位性能问题时，")
    print("  忘了这一步会得出完全错误的结论。")


# ---------------------------------------------------------------------------
banner("动手改改看")
# ---------------------------------------------------------------------------

print("""
1. 第 4 节里把 m.t() 换成 m[:, ::2]（隔列取），看看 stride 变成什么，
   is_contiguous 还是不是 True。

2. 第 8 节里把矩阵从 4096 改成 256，再看两种计时的差距。
   矩阵变小之后差距是变大还是变小？为什么？

3. 第 2 节里试着让 float32 和 bfloat16 相乘成功，需要加哪一步？
""")
