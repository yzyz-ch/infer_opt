# Stage 0 代码导读

读代码时用这份笔记定位：每个文件在整条生成链上负责哪一步。怎么跑见同目录 [`guide.md`](guide.md)；实验数字和自测题见 [`manual_decode_notes.md`](manual_decode_notes.md)。

---

## 1. 这个阶段学什么

阶段 0 把 `model.generate()` 拆开，自己写出里面的最小生成循环。模型仍由 Hugging Face 加载，Qwen3-0.6B 的 Transformer 结构这一阶段不重写。

要亲手走通的路径是：

```text
用户问题
  → apply_chat_template 套上对话格式
  → tokenizer 变成 input_ids          形状 [batch, seq_len]
  → model(...) 一次 forward
  → logits                            形状 [batch, seq_len, vocab]
  → 只取最后一个位置 [:, -1, :]
  → argmax 选出下一个 token id
  → 拼回输入，再 forward
  → 遇到 EOS 或达到 max_new_tokens 停止
```

五组脚本沿着这条链往下加一层：

| 目录 | 加的那一层 | 读完应能回答 |
| --- | --- | --- |
| `00a_tensor_basics/` | PyTorch 地基，不加载大模型 | shape / dtype / device 为什么必须对上；转置为什么不搬数据；GPU 计时为什么要同步 |
| `00b_one_forward/` | 只做一次 forward | forward 吃什么、吐什么；为什么生成时只用最后一行 logits |
| `00c_manual_decode/` | 把这一步重复 N 遍，故意不用 KV Cache | 一次 forward 只产出一个 token；短序列上看不到变慢 |
| `00d_kv_cache/` | 给循环装上 KV Cache | 缓存的是什么；什么长度下才真的省时间 |
| `00e_compare_with_generate/` | 和官方 `generate(do_sample=False)` 逐 token 对齐 | `generate()` 替循环做了哪些事；文字不同不等于写错 |

三条贯穿全阶段的结论，后面读优化代码时会反复用到：

1. **一次 forward 只产出一个新 token。** 生成 N 个词就要 forward N 次。后面的 KV Cache、continuous batching、投机解码，都是在让这个重复过程变便宜。
2. **优化有生效范围。** KV Cache 在很短的上下文里加速比接近 1，上下文拉长后才明显。性能结论必须带着序列长度一起说。
3. **两个都正确的实现，bf16 下文字可以不同。** 矩阵形状一变，kernel 和累加顺序就变，logits 有微小抖动。前两名咬得很紧时 argmax 会翻转，后面整段话都岔开。对齐时比 token 路径，不要只比最终文字。

训练侧这一阶段只需要记住四点，代码里没有反向传播：模型练的是「根据前面的 token 预测下一个」；因果遮罩让当前位置看不到未来；推理只读已经训好的权重；推理前要 `eval()` 并关掉梯度记录。

---

## 2. 目录怎么串起来

`stage0/` 根上的 `00a_tensor_basics.py` … `00e_compare_with_generate.py` 不是逻辑本体。它们用 `runpy` 转去跑对应目录里的 `run_all.py`。真正要读的是各子目录里按编号命名的小文件。

子目录里的 `_path.py` 把 `stage0/` 插进 `sys.path`，这样小脚本能 `import common`。单独跑某个编号文件时，它的 `if __name__ == "__main__"` 会先把前面的步骤再执行一遍（所以会重新加载模型）。`run_all.py` 则只加载一次，把返回值传给后面的 `run()`。

`00e_compare_with_generate/loops.py` 用 `importlib` 按文件路径加载 `00c` 的 `greedy_loop.py` 和 `00d` 的 `cached_loop.py`。根目录已经有同名的 `00c_manual_decode.py`、`00d_kv_cache.py` 入口，不能靠普通 `import` 去拿解码函数。

读代码的顺序就是编号顺序：`common.py` → `00a` 的 01～08 → `00b` 的 01～06 → `00c` 的循环本体再读测量脚本 → `00d` → `00e`。

---

## 3. `common.py`：每个脚本都要的样板

这个文件不单独运行。它把「加载、造 prompt、取 EOS、计时」从 00b～00e 里抽出来，让各脚本只留下自己要演示的那几行。

| 名字 | 做什么 |
| --- | --- |
| `MODEL_PATH` | 指向仓库根下的 `local/models/Qwen3-0.6B`，不写本机绝对路径 |
| `DEVICE` / `DTYPE` | 有 CUDA 用 GPU + bfloat16；否则 CPU + float32。bf16 显存是 fp32 的一半，指数位和 fp32 一样宽 |
| `load_model_and_tokenizer()` | `AutoTokenizer.from_pretrained` 读分词规则；`AutoModelForCausalLM.from_pretrained(..., dtype=DTYPE)` 读 config、搭网络、填 `model.safetensors`；`model.to(DEVICE)` 搬到 GPU；`model.eval()` 关掉训练期随机行为。参数名是 transformers v5 的 `dtype`，旧教程里的 `torch_dtype` 已弃用 |
| `DEFAULT_QUESTION` | 全阶段固定一句「用一句话说明什么是大语言模型推理。」，方便横向对比 |
| `build_prompt()` | 用 `apply_chat_template` 把问题套成 `<\|im_start\|>user ... <\|im_end\|>\n<\|im_start\|>assistant\n`。`tokenize=False` 先要文本，方便打印。`add_generation_prompt=True` 在末尾补上 assistant 开头。`enable_thinking=False` 关掉思考链，避免输出几百个 `<think>` token |
| `get_eos_ids()` | 从 `generation_config.eos_token_id` 收成 `set`。Qwen3 这里是列表 `[151645, 151643]`（`<\|im_end\|>` 和 `<\|endoftext\|>`），和 `config.json`、tokenizer 里的单个整数不是同一份配置。停止条件必须用 `in` |
| `cuda_sync()` / `Timer` | 进出计时区间都 `torch.cuda.synchronize()`。CUDA 异步，Python 返回只表示任务进了队列 |

裸问题直接喂给这个指令模型，它会当成续写，很久都不吐 EOS。阶段 0 所有生成都走 `build_prompt()`。

---

## 4. `00a_tensor_basics/`：不加载大模型的地基

八个脚本用小 tensor 把后面代码里会碰到的 PyTorch 概念跑一遍。`util.py` 只有一个 `cuda_sync()`，给 `08` 用。

### `01_shape.py`

`torch.zeros(2, 3, 4)` 看 `shape`、`ndim`、`numel()`。`view(2, 12)` 元素总数不变，只换看待方式。后面反复出现的三种形状在这里先记名字：

- `input_ids`：`[batch, seq_len]`
- `hidden`：`[batch, seq_len, hidden_size]`
- `logits`：`[batch, seq_len, vocab_size]`

### `02_dtype.py`

同一块 `1024×1024` 分别用 fp32 / fp16 / bf16 / int8 创建，用 `element_size()` 算占用。fp32 和 bf16 直接做矩阵乘会报错，要先 `.to()` 成同一种 dtype。推理几乎用 bf16：显存和访存都减半，而大模型推理的瓶颈通常是把权重从显存读出来。

### `03_device.py`

默认 tensor 在 CPU。`.to("cuda")` 返回新 tensor，不接返回值等于没搬。CPU tensor 和 CUDA tensor 不能直接相加。所以后面每个脚本都把 `input_ids` 放到 `model.device`：权重在 GPU 上，输入留在 CPU，forward 第一步就崩。

### `04_stride_contiguous.py`

`m.t()` 之后 `shape` 和 `stride` 变了，`is_contiguous()` 变成 False，但 `untyped_storage().data_ptr()` 和原来相同：转置只改了「怎么读这块内存」的说明书，没有复制数据。`contiguous()` 才会另拷一份，地址才变。

`m[:, ::2]` 同样共享内存，列方向 stride 变成 2，中间有洞，也不是 contiguous。PyTorch 切片的 step 必须大于 0，倒序用 `flip`。

阶段 7 写 CUDA kernel 时如果只检查 shape，转置或切片过来的 tensor 形状看着对，读出来的数是错位的，而且不报错。

### `05_nn_module_state_dict.py`

最小的 `TinyNet`：两层 `nn.Linear`，`forward` 里 `fc2(relu(fc1(x)))`。打印模块就是结构树。`state_dict()` 是「参数名 → tensor」的字典，例如 `fc1.weight`。Qwen3 的 `model.safetensors` 是同一类字典，只是条目多得多。阶段 2 做模型体检时会逐项核对这些名字和形状。

### `06_eval.py`

带 `Dropout(p=0.5)` 的小网络，同样输入跑三次：`train()` 下每次结果不同，因为随机丢神经元；`eval()` 下三次完全一样。Qwen3 推理路径上其实没有 Dropout，但 `load_model_and_tokenizer()` 里仍然调用 `model.eval()`，养成推理前切到确定性模式的习惯。

### `07_inference_mode.py`

`requires_grad=True` 的 tensor 做运算时，普通模式会挂上 `grad_fn`，记下反向传播需要的计算图。包在 `torch.inference_mode()` 里则 `requires_grad=False`、`grad_fn=None`。推理不更新权重，这些记录是多余的显存和开销。`no_grad()` 效果相近，`inference_mode()` 更彻底。00b 往后的 forward 和生成循环都包在这个上下文里。

### `08_cuda_timing.py`

同一段 `4096×4096` 矩阵乘测十次：先预热，再对比「循环结束立刻停表」和「停表前 `synchronize()`」。不同步量到的是把任务丢进队列的时间。后面所有性能数字都先预热、再同步。预热还要覆盖真正会用到的矩阵形状，见 `00d` 的 warmup。

---

## 5. `00b_one_forward/`：一次 forward，不循环

00c 的生成循环就是把这里重复 N 遍。各文件的 `run()` 接收上一步的返回值；单独运行时 `__main__` 会把 01、02… 串起来。

### `01_load.py`

调用 `load_model_and_tokenizer()`，有 GPU 时打印 `torch.cuda.memory_allocated()`。权重占用大约是「参数量 × 2 字节」。返回 `(tokenizer, model)`。

### `02_tokenize.py`

先打印问题原文，再打印 `build_prompt()` 之后的文本，确认特殊 token 已经套上。然后：

```text
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.device)
```

`input_ids` 是整数，形状 `[1, seq_len]`。循环前 25 个 id，用 `tokenizer.decode([tid])` 看每个 id 对应哪一小段文本。

文字在 `tokenizer(...)` 这一步结束。之后模型内部只有整数 id 和浮点激活。

### `03_one_forward.py`

在 `inference_mode()` 里对同一份 `input_ids` forward 两次，用 `Timer` 分别记冷启动和第二次。第一次含 kernel 选择等预热，明显更慢，所以性能测量都要先跑一遍再计时。

返回对象是 transformers 的输出，字段里有 `logits`。默认形状 `[1, seq_len, vocab_size]`：batch 里这一条序列的每个位置，都对词表里每个词打了一个分。脚本会算出这坨浮点数占多少 MB，下一节说明其中绝大部分生成时用不上。

### `04_last_position.py`

对位置 0、1、中间、倒数第二、最后一个，分别取该位置的输入词，以及对 `logits[0, pos]` 做 `argmax` 得到的「下一个词」。

因果语言模型在位置 `t` 预测的是位置 `t+1`。前面那些位置预测的词，输入里已经有了。只有最后一个位置指向还没生成的词，所以生成时取 `logits[:, -1, :]`。

接着用 `logits_to_keep=1` 再 forward 一次。lm_head 只保留最后一行，输出变成 `[1, 1, vocab_size]`，省掉绝大部分 lm_head 计算和输出显存。`generate()` 内部就是这样。阶段 0 的 00b/00c 仍用默认全量 logits，是为了把每一行都打印出来看；00d 的缓存循环已经改成 `logits_to_keep=1`。

同一行两种算法在 bf16 下不一定 `torch.equal`：矩阵形状变了会走不同 kernel，数值有微小差，argmax 往往仍一致。后面对比比的是 token id。

### `05_next_token.py`

`last_logits = logits[0, -1, :]`，`argmax` 得到下一个 token id，再 `decode` 成文本。另外把 logits 转成 fp32 做 `softmax`，`topk` 打印概率最高的 5 个候选。阶段 0 只用 argmax（greedy），没有随机性，才能和 `generate(do_sample=False)` 逐 token 比。

### `06_summary.py`

把上面收成一句话：输入 L 个 token，一次 forward 得到 1 个新 token；要第 2 个，就拼回去变成 L+1 再跑。文件末尾是三道练习：打开 `enable_thinking`、跳过 chat template、把 argmax 换成 `multinomial` 抽样。

---

## 6. `00c_manual_decode/`：手写循环，故意不用缓存

### `greedy_loop.py`（循环本体，被 00e 复用）

`manual_greedy_decode()` 是这一阶段第一版生成器。`MAX_NEW_TOKENS = 64`。

每一步做的事：

1. 第一次进入循环前，整段 prompt 已经变成 `input_ids`。
2. `model(input_ids=input_ids)`，整条序列重新前向，不传 `past_key_values`。
3. `logits[0, -1, :]` 再 `argmax`，得到 `next_id`。
4. `torch.cat` 把形状 `[1, 1]` 的新 token 接到序列末尾，作为下一轮的完整输入。
5. 记下这一步的输入长度、logits 形状、token id、耗时。
6. `next_id in eos_ids` 则 break。`for/else`：循环没被 break、而是跑满上限时，打印「强制截断」。

返回的 `generated_ids` 不含 prompt，只有新生成的 id。每一步的输入长度比上一步多 1，logits 的 seq 维也跟着变长。

### `01_run_loop.py`

加载模型、造 prompt，先用 prompt 的形状做一次不记时的 forward 当预热，再调用 `manual_greedy_decode` 把每步表格打出来。

### `02_show_result.py`

`tokenizer.decode(generated_ids, skip_special_tokens=True)` 得到可读文本；`skip_special_tokens=False` 能看到结尾的 `<|im_end|>`。结束符是循环决定停下来的，decode 本身不会因为见到它就截断，除非你让它跳过特殊 token。

### `03_step_timing.py`

用 `records` 里的 `elapsed_ms` 看前几步和后几步。prompt 只有二十来个 token、生成也不长时，每步耗时几乎是平的。这段长度上 Attention 的计算量很小；每步却要把约 1.1GB 权重读一遍，还要启动 28 层里的一大批 kernel。固定开销盖住了「序列变长应该变慢」。

### `04_length_sweep.py`

用随机 token id 造长度 16、64、256、512、1024、2048、4096 的输入。每个长度预热 3 次，再计时 10 次平均，forward 时带 `logits_to_keep=1`，避免 lm_head 把超长序列的全词表输出也算进时间。

512 以内基本是水平线，固定开销主导。1024 往上才爬升。长度再翻倍时耗时涨得比线性更多，因为 Attention 要让每个 token 和前面所有 token 做相关，计算量按序列长度的平方增长。

### `05_where_waste.py`

把每步的 `input_len` 加起来，和步数比。N 步里真正新的 token 只有 N 个，其余都是把上一轮已经算过的前缀再算一遍。重复计算的比例可以很高，但在当前这么短的序列上，浪费的时间几乎看不出来，因为瓶颈在读权重和启动 kernel。

每个 token 的 Key / Value 只取决于它自己和它前面的内容，下一轮不会变。存下来复用就是 KV Cache，交给 `00d`。

---

## 7. `00d_kv_cache/`：同一条循环，加上缓存

### `cached_loop.py`

`cached_greedy_decode()` 和 00c 的差别在「每轮喂什么」和「K/V 放哪」。

- 进入循环前建一个空的 `DynamicCache()`。transformers v5 的 `past_key_values` 要这个对象；v4 教程里的嵌套 tuple 在这里会报错。
- **第 0 步是 Prefill**：把整段 prompt 喂进去，`use_cache=True`，`logits_to_keep=1`。模型算出每个位置的 K/V 写入 cache，并给出最后一个位置的 logits。`cache.get_seq_length()` 变成 prompt 长度。
- 若第一个新 token 不是 EOS，**之后每步是 Decode**：`input_ids` 只有刚选出的那 1 个 token，同一份 cache 传回去。模型只为这个新 token 算 Q、K、V，K/V 追加到缓存末尾，注意力去读缓存里已有的历史。
- 停止条件仍是 `next_id in eos_ids`，或生成个数达到上限。

`records` 里多了 `fed`（本轮喂入几个 token）、`cache_len`、`phase`（`prefill` 或 `decode`）。

`benchmark(model, context_len, n_new=32)` 用随机 token 造固定上下文，各生成 32 个 token，先各跑一遍预热，再计时。`run_no_cache` 每步 `cat` 整段再 forward；`run_cache` 是上面的 prefill + 只喂 1 token。返回 `(无缓存秒数, 有缓存秒数)`。

`kv_cache_bytes_per_token()` 按配置算一个 token 的缓存大小：

```text
2 × num_hidden_layers × num_key_value_heads × head_dim × 每元素字节数
```

前面的 2 是 Key 和 Value 各一份。Qwen3-0.6B 是 GQA：注意力头多于 KV 头，缓存按 KV 头数算，不按全部注意力头算。`torch.finfo(model.dtype).bits // 8` 在 bf16 下是 2 字节。

### `01_cached_loop.py`

预热故意跑两种形状：先整段 prompt（prefill），再一个 token（decode）。只预热 prefill 的话，第一次 decode 会因为 `[1, 1]` 这个新形状重新挑 kernel，计时出现尖峰。然后调用 `cached_greedy_decode` 并打印去掉特殊 token 的文本。

### `02_prefill_vs_decode.py`

用 `records[0]` 当 prefill，其余当 decode，打印喂入 token 数和耗时。Prefill 一次处理整段 prompt，decode 每次 1 个 token，两者墙钟时间却往往接近：两步都要把全部权重从显存读一遍，读权重的时间和喂几个 token 关系不大。Prefill 把这次读权重摊到很多 token 上；decode 是为了 1 个 token 把权重读完，所以称为访存受限。batch 能让一次读权重服务多条请求，这是后面 continuous batching 要做的事。

同一节用上面的公式把「本次缓存多少 MB」「8192 上下文一条请求多少 MB」「16GB 卡去掉权重大概能同时放几条」算出来。KV Cache 是用显存换「不用重算历史 K/V」。显存怎么切块分配，是阶段 4 PagedAttention 的题目。

### `03_when_faster.py`

对上下文 128、512、1024、2048、4096 各调一次 `benchmark`，打印加速比 `t_no / t_yes`。短上下文接近 1 倍甚至略低于 1：缓存帮不上读权重和启动 kernel，还多了维护 cache 的开销。上下文变长后，「整段重算」随长度变贵，「用缓存」的每步代价基本不动，加速比才上去。这一节最慢，`run_all.py` 会跑它。

### `04_bf16_and_exercises.py`

解释 00c 和 00d 文字可能不同：00c 每步矩阵是「当前全长 × hidden」，00d 的 decode 是「1 × hidden」。形状不同，cuBLAS 选不同 kernel，bf16 尾数只有 8 位，累加顺序的差别足以让 logits 差零点几。某一步第一名和第二名贴得很近时 argmax 翻转，自回归会把后面整段带跑。验证要让数值路径一致，或者看 top-1 一致率，而不是要求两段文字逐字相同。00e 把这件事量出来。

---

## 8. `00e_compare_with_generate/`：三种实现对齐

### `loops.py`

按路径加载 00c / 00d 的解码函数，返回两个模块。见第 2 节。

### `01_run_three.py`

同一条 prompt、同一个 `MAX_NEW_TOKENS = 64`，跑三次：

1. `manual_greedy_decode`（无缓存）
2. `cached_greedy_decode`（有缓存）
3. `model.generate(..., do_sample=False, max_new_tokens=64)`，并传入全 1 的 `attention_mask`

`generate` 的输出含 prompt，用 `out[0, prompt_len:]` 切掉，只留新 token，才能和前两种的 `generated_ids` 比。

`generation_config.json` 默认 `do_sample: true`，还写了 temperature / top_p / top_k。这里显式关掉采样后，那些参数用不上，transformers 会警告，属预期。

### `02_compare.py`

`first_diff` 按位置找第一处不同的 token id；长度不同则分歧点算在较短的那个末尾。打印三段文本，以及无缓存、有缓存各自相对 `generate()` 从第几步分叉。

`generate()` 默认开着 KV Cache，和 00d 走同一条数值路径，所以有缓存的手写循环可以逐 token 一致。无缓存那版逻辑也是对的，只是矩阵形状不同，允许在某一步分叉。

### `03_inspect_fork.py`

若 `02` 没找到分歧就跳过（换个问题容易复现）。否则取分歧步 `k` 之前的共同前缀，拼回 prompt，再跑两条路径：

- 路径 A：整段 `prefix` 一次 forward，取最后一行 logits（对应 00c）
- 路径 B：先用 `prefix` 去掉最后一个 token 做一次带 cache 的 forward，再用最后 1 个 token 做 decode（对应 00d）

两边各 `topk(3)`，并比较「第一名和第二名的分差」和「两条 logits 的最大绝对差」。分差小于数值误差时，谁排第一由舍入决定。

### `04_wrong_eos.py`

并排打印三处 `eos_token_id`：`model.config`、`generation_config`、`tokenizer`。生效的是 `generation_config` 里的列表。

然后故意写 `if nid == model.generation_config.eos_token_id`。int 和 list 比较恒为 False，循环不会因 EOS 停下。脚本仍用正确的 `eos_ids` 记录「本该在第几步停」，用来对照实际跑满的步数。越过 `<|im_end|>` 之后模型会把它当成普通 token，按对话格式继续往下写，甚至开始新一轮 user。停不停是外层循环的事。

### `05_generate_and_checklist.py`

把 `generate()` 藏起来的工作收成 9 条。阶段 0 写过的是前 6 条：

1. 循环，直到停止
2. 从 logits 里取最后一个位置
3. 选词（这里是 greedy；采样、beam 还没写）
4. 把新 token 接回上下文（无缓存时是拼进 `input_ids`；有缓存时是交给 cache，下一轮只喂新 id）
5. EOS 集合和 `max_new_tokens`
6. 建立 cache、区分 prefill / decode

还没写的是 attention_mask / position_ids 的细维护、变长 batch 的 padding、以及从 `generation_config` 读采样参数。文件末尾的自查清单和 `manual_decode_notes.md` 里的问答题是同一组问题。

---

## 9. 读某段代码时先对哪

| 你在代码里看到 | 它在干什么 | 先看 |
| --- | --- | --- |
| `apply_chat_template` | 把一句问话变成模型训练时见过的对话文本 | `common.build_prompt` |
| `tokenizer(..., return_tensors="pt")` | 文本 → `input_ids`，还在 CPU 上，除非接着 `.to()` | `00b/02_tokenize.py` |
| `model.eval()` / `inference_mode()` | 关掉 Dropout 一类随机行为；不记梯度 | `00a/06`、`00a/07`，调用点在 `common.py` |
| `outputs.logits` 的三维形状 | 每个位置对整个词表的分数 | `00b/03_one_forward.py` |
| `logits[0, -1, :]` 或 `logits_to_keep=1` | 只要「下一个词」，丢掉对已知位置的预测 | `00b/04_last_position.py` |
| `.argmax(dim=-1)` | greedy，取得分最高的 token id | `00b/05_next_token.py` |
| `torch.cat([input_ids, next_token], dim=1)` | 无缓存时把新 token 接回，下一轮整段重算 | `00c/greedy_loop.py` |
| `next_id in eos_ids` | 用结束符集合停循环 | `common.get_eos_ids`，错误写法在 `00e/04_wrong_eos.py` |
| `DynamicCache` + `past_key_values` + `use_cache=True` | 历史 K/V 留在缓存里 | `00d/cached_loop.py` |
| prefill 喂整段、decode 只喂 `[[next_id]]` | 两种阶段的输入长度不同 | `cached_greedy_decode` 里 step 0 和后面的 for |
| `Timer` / `cuda_sync` | 量的是 GPU 算完的时间 | `00a/08_cuda_timing.py`、`common.Timer` |
| 无缓存对不上 `generate()`，有缓存能对上 | 数值路径相同才会逐 token 一致 | `00e/02_compare.py`、`03_inspect_fork.py` |
