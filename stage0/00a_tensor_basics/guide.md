# 00a：推理需要的 PyTorch 地基

不加载大模型，用小 tensor 把概念一个一个跑通。建议按编号顺序读、按编号顺序跑。

两个反直觉实验别跳过：

- `04_stride_contiguous.py`：转置之后数据其实没有搬动
- `08_cuda_timing.py`：不同步就计时会得到假数字

## 怎么跑

先进入 `stage0` 并激活环境，再逐个运行：

```bash
cd stage0
source ../local/venvs/.venv-hf/bin/activate

python 00a_tensor_basics/01_shape.py
python 00a_tensor_basics/02_dtype.py
python 00a_tensor_basics/03_device.py
python 00a_tensor_basics/04_stride_contiguous.py
python 00a_tensor_basics/05_nn_module_state_dict.py
python 00a_tensor_basics/06_eval.py
python 00a_tensor_basics/07_inference_mode.py
python 00a_tensor_basics/08_cuda_timing.py
```

也可以一次跑完全部：

```bash
python 00a_tensor_basics/run_all.py
```

`util.py` 只提供标题打印和 GPU 同步，不用单独运行。

## 每个文件在讲什么

| 文件 | 主题 |
| --- | --- |
| `01_shape.py` | tensor 每个维度有多大 |
| `02_dtype.py` | 每个数字用什么格式存 |
| `03_device.py` | 数据放在哪块硬件上 |
| `04_stride_contiguous.py` | stride、contiguous、转置不搬数据 |
| `05_nn_module_state_dict.py` | `nn.Module` 和 `state_dict` |
| `06_eval.py` | 推理前为什么要 `eval()` |
| `07_inference_mode.py` | 为什么推理要关掉梯度记录 |
| `08_cuda_timing.py` | CUDA 异步与正确计时 |
