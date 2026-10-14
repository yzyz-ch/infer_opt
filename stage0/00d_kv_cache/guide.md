# 00d：给手写循环装上 KV Cache

短对话里几乎不省时间；长上下文才能明显加速。用数字量出这条分界线。

## 怎么跑

```bash
cd stage0
source ../local/venvs/.venv-hf/bin/activate
python 00d_kv_cache/01_cached_loop.py
python 00d_kv_cache/run_all.py
```

`03_when_faster.py` 最慢，可先跳过，读完 01、02、04 再回来跑。

## 每个文件

| 文件 | 主题 |
| --- | --- |
| `cached_loop.py` | 带缓存的循环和 benchmark（被 00e 复用） |
| `01_cached_loop.py` | Prefill 一次，之后每步只喂 1 个 token |
| `02_prefill_vs_decode.py` | 两种阶段的计算特征，以及缓存占多少显存 |
| `03_when_faster.py` | 不同上下文长度下的加速比 |
| `04_bf16_and_exercises.py` | 为何和 00c 文字可能不同，以及练习 |
