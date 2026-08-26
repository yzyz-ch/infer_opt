# 00e：和 model.generate() 对齐

能跑出通顺句子不等于写对了。必须和官方实现逐 token 比对。

## 怎么跑

```bash
cd stage0
source ../local/venvs/.venv-hf/bin/activate
python 00e_compare_with_generate/run_all.py
```

`01_run_three.py` 会加载模型并跑三种实现，单独跑 02～04 时也会先执行 01。

## 每个文件

| 文件 | 主题 |
| --- | --- |
| `loops.py` | 加载 00c / 00d 的解码函数 |
| `01_run_three.py` | 无缓存、有缓存、`generate()` 各跑一遍 |
| `02_compare.py` | 三者逐 token 对比 |
| `03_inspect_fork.py` | 分歧那一步的 logits 拆开看 |
| `04_wrong_eos.py` | EOS 写成 `==` 会停不下来 |
| `05_generate_and_checklist.py` | `generate()` 藏了什么，以及自查清单 |
