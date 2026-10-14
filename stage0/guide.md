# 阶段 0：不使用 `generate()`，自己写最小生成循环

对应学习计划第 4 节。目标是把 `model.generate()` 这层外壳拆开，
亲手实现里面的生成循环。

模型仍然由 Hugging Face 加载——**这一阶段不重写 Transformer**，
只写 `generate()` 隐藏起来的那部分逻辑。重写模型是阶段 2 和阶段 7 的事。

## 环境

需要 NVIDIA GPU、已安装的 Hugging Face 环境，以及本地的 Qwen3-0.6B 权重
（默认路径 `../local/models/Qwen3-0.6B`）。

```bash
source ../local/venvs/.venv-hf/bin/activate
cd stage0
```

transformers 若是 **v5**，和网上多数 v4 教程有差异：
`past_key_values` 已经从元组换成了 `Cache` 对象。

## 按顺序运行

```bash
python 00a_tensor_basics/01_shape.py              # 地基 01～08
python 00b_one_forward/02_tokenize.py             # 一次 forward
python 00c_manual_decode/01_run_loop.py           # 手写循环
python 00d_kv_cache/01_cached_loop.py             # KV Cache
python 00e_compare_with_generate/run_all.py       # 与 generate() 对齐
```

每个主题都拆成了独立小文件，目录里有 `guide.md`。想一次跑完某一主题：

```bash
python 00a_tensor_basics/run_all.py
python 00b_one_forward/run_all.py
python 00c_manual_decode/run_all.py
python 00d_kv_cache/run_all.py          # 含长序列加速比，约 30 秒
python 00e_compare_with_generate/run_all.py
```

必须先 `cd` 到本目录再运行：脚本要 `import common`。

不想激活环境的话，也可以直接用解释器路径调用：

```bash
../local/venvs/.venv-hf/bin/python 00b_one_forward/run_all.py
```

## 每个脚本在讲什么

| 脚本 | 主题 | 跑完你应该能回答 |
| --- | --- | --- |
| `00a_tensor_basics/` | PyTorch 地基（01～08） | 转置为什么不搬数据？GPU 计时为什么必须同步？ |
| `00b_one_forward/` | 一次 forward | forward 吃什么吐什么？为什么只用最后一行 logits？ |
| `00c_manual_decode/` | 手写生成循环 | 一次 forward 只产出一个 token 意味着什么？ |
| `00d_kv_cache/` | KV Cache | 缓存的是什么？什么条件下它才真的加速？ |
| `00e_compare_with_generate/` | 与官方对齐 | `generate()` 到底替我做了哪些事？ |

`common.py` 是公共工具（模型路径、加载、prompt 构造、计时），不单独运行。

## 三个关键结论

跑完之后，这三条是最值得记住的：

**1. 一次 forward 只产出一个 token。**
生成 30 个 token 就要 forward 30 次。整个推理优化领域，
本质上都在想办法让这个重复过程变便宜。

**2. 优化都有生效范围，必须先量再用。**
KV Cache 在 128 token 上下文里加速比约 1.0x（等于没用），
到 4096 上下文才涨到 6 倍左右。脱离条件谈"某某优化更快"没有意义。

**3. 两个都正确的实现，输出可以不一样。**
bf16 精度下，不同矩阵形状走不同 kernel，logits 有 0.0x 量级抖动。
某一步前两名咬得紧时 argmax 会翻转，之后整段话都不同。
验证正确性不能只看"文字一不一样"。

## 已知的坑

- `eos_token_id` 在 `generation_config.json` 里是**列表** `[151645, 151643]`，
  而 `config.json` 和 tokenizer 里都是单个整数 `151645`。三处不一致，
  真正生效的是 generation_config。用 `in` 判断，不要用 `==`。
- `generation_config.json` 默认 `do_sample: true`，对比时必须显式传 `do_sample=False`。
- transformers v5 的 `past_key_values` 必须传 `Cache` 对象，v4 的元组写法会报错。
- 带 KV Cache 时 decode 阶段每轮只喂新 token，不要再喂整条序列。
- GPU 计时前必须 `torch.cuda.synchronize()` 并预热，否则测出来的数字没有意义。
  预热要覆盖所有会用到的矩阵形状（prefill 和 decode 是两种形状）。
- 裸 prompt 喂给 Qwen3-0.6B 会一直复读停不下来，必须套 chat template。
  阶段 0 用 `enable_thinking=False` 关掉思考链，输出更短、便于观察。

## 下一步

- 填写 `manual_decode_notes.md`
- 对照 `00e` 最后的自查清单，确认每一条都能不看代码讲清楚
- 进入阶段 1：用三种方式跑通同一个小模型
