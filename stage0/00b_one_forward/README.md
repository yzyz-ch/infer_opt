# 00b：只做一次 forward，不循环

把「模型 forward 的输入和输出到底是什么」看清楚。想明白之后，00c 的生成循环只是把它重复 N 遍。

## 怎么跑

```bash
cd stage0
source ../local/venvs/.venv-hf/bin/activate
python 00b_one_forward/01_load.py
python 00b_one_forward/02_tokenize.py
# ...
python 00b_one_forward/run_all.py    # 只加载一次模型，按顺序跑完全部
```

单独跑 03 及以后的文件时，会先加载模型并执行前面的步骤。

## 每个文件

| 文件 | 主题 |
| --- | --- |
| `01_load.py` | 加载模型和 tokenizer |
| `02_tokenize.py` | 文本在哪一步变成 token id |
| `03_one_forward.py` | forward 吃什么、吐什么 |
| `04_last_position.py` | 为什么只用最后一行 logits |
| `05_next_token.py` | logits 怎么变成下一个词 |
| `06_summary.py` | 小结和练习 |
