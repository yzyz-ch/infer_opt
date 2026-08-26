# 00c：手写生成循环

不调用 `model.generate()`，自己写 greedy 循环。这一版故意不用 KV Cache。

## 怎么跑

```bash
cd stage0
source ../local/venvs/.venv-hf/bin/activate
python 00c_manual_decode/01_run_loop.py
python 00c_manual_decode/run_all.py
```

`04_length_sweep.py` 会扫较长序列，单独跑大约十几秒。

## 每个文件

| 文件 | 主题 |
| --- | --- |
| `greedy_loop.py` | 循环本体（被 00e 复用，不单独跑） |
| `01_run_loop.py` | 跑通循环，逐步打印 |
| `02_show_result.py` | 解码成文字 |
| `03_step_timing.py` | 短序列上耗时为什么几乎不变 |
| `04_length_sweep.py` | 把序列拉长后才能看到变慢 |
| `05_where_waste.py` | 重复计算在哪，以及练习 |
