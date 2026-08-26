# LLM 推理优化工程师学习计划（通俗版）

## 0. 这份计划是为谁写的

这份计划针对下面这种情况：

- 已经做过 GPU 算子开发，熟悉 C++、CUDA、访存和性能优化。
- 熟悉 Attention 的计算过程，也了解 MHA、GQA、MQA、MLA 等结构。
- 知道 Transformer 大致由哪些模块组成。
- PyTorch 只做过简单使用。
- 没有训练过模型，也没有完整部署过模型。
- 希望从“单个算子优化”扩展到“整个模型推理过程的优化”。
- 长期工作方向偏国产 GPU、新硬件适配和推理 Runtime 性能优化。

这里的目标不是培养模型训练工程师，也不是培养 Kubernetes 运维工程师。

最终希望形成的能力是：

```text
看到一个模型推理慢或运行失败
→ 知道问题可能属于哪一层
→ 能沿着请求路径找到对应代码
→ 能判断是模型加载、框架调度、显存、算子、通信还是服务层的问题
→ 能设计最小实验验证判断
→ 能修改代码
→ 能证明修改后结果仍然正确，而且端到端确实有收益
```

---

## 1. 先把最容易混淆的词说清楚

后面的计划会使用专业词，但每个词都会尽量在第一次出现时解释。

### 1.1 模型是什么

可以先把模型理解为：

```text
一套计算结构 + 很多已经调整好的数字
```

计算结构包括 Attention、MLP、RMSNorm、Embedding 等模块。

那些已经调整好的数字叫做“权重”。权重一般保存在 `safetensors` 文件中，本质上是很多 tensor。

模型文件本身不会自动回答问题。需要有程序把权重读进内存，再按照模型结构执行计算。

### 1.2 训练是什么

训练是通过大量数据不断修改权重，让模型逐渐学会预测下一个 token。

训练通常包含：

```text
前向计算
→ 计算答案错了多少
→ 反向传播计算梯度
→ 优化器修改权重
```

这份计划不要求你实现完整训练。你只需要理解权重是怎么来的，以及训练和推理的边界。

### 1.3 推理是什么

推理是使用已经训练好的权重产生结果。

推理时通常不修改权重，也不做反向传播。

例如：

```text
用户输入一句话
→ 转成 token id
→ 模型计算下一个 token 的分数
→ 选择一个 token
→ 重复执行
→ 返回一段文字
```

### 1.4 部署是什么

部署是让模型在一台或多台机器上真正运行起来，并且可以被程序或用户调用。

最小部署只需要：模型能加载、输入能进入、输出能返回。

生产部署还要处理并发、异常、监控、限流、扩容和稳定性。

### 1.5 Token 是什么

模型不能直接读取中文字符串。文本会先被 tokenizer 切分成 token，再转换成整数编号，也就是 token id。

例如：

```text
“你好”
→ tokenizer
→ [108386]
```

具体结果由模型使用的 tokenizer 决定。

### 1.6 Runtime 是什么

Runtime 可以理解成“模型推理时的总调度中心”。

它不负责训练模型，主要负责：

```text
接收请求
管理请求状态
组织 batch
管理 KV Cache
调用模型执行
调用底层算子
采样下一个 token
判断请求是否结束
```

vLLM 和 nano-vLLM 中的 Engine、Scheduler、ModelRunner 等模块都属于 Runtime 的一部分。

### 1.7 Backend 是什么

Backend 在不同语境中含义不完全一样。这份计划中的“推理后端”主要指：

```text
推理框架和具体硬件之间的执行层
```

它需要回答：

- tensor 怎么放到这种设备上？
- 使用哪种显存分配器？
- stream 和 event 怎么工作？
- Attention、GEMM、RMSNorm 调哪个实现？
- 图执行怎么捕获和重放？
- 多卡时使用什么通信库？
- 用什么工具做性能分析？

国产 GPU 适配通常会大量涉及这一层。

### 1.8 Kernel 是什么

Kernel 就是在 GPU 上真正执行的一段计算程序，例如 GEMM、Attention、RMSNorm 或 RoPE。

你原来主要优化的是这一层。

这份计划要帮助你进一步理解：是谁准备 kernel 的输入，谁选择这个 kernel，谁调用它，以及它变快后整个模型是否真的变快。

### 1.9 Serving 是什么

Serving 是把模型变成一个长期运行、可以接收外部请求的服务。

它比 Runtime 更靠近用户：

```text
用户/业务程序
→ HTTP 服务
→ 请求排队与流式返回
→ Runtime
→ 模型与 kernel
```

这份计划只学习能帮助端到端性能分析的 Serving 知识，不会把 Kubernetes、网关和大规模集群运维当作当前主线。

### 1.10 用一个饭店类比整套系统

这个类比只帮助建立第一印象，不代表所有技术细节完全相同。

```text
用户请求          = 顾客点菜
Serving           = 前台接单、排队、返回结果
Runtime           = 后厨总调度
Scheduler         = 决定哪几道菜先做
KV Cache          = 保存已经准备好的中间材料
ModelRunner       = 组织一次实际烹饪过程
Kernel            = 厨师执行具体动作
GPU               = 炉灶和厨房设备
Profiler          = 摄像和计时，找出到底慢在哪里
```

---

## 2. 一次 LLM 请求的大致路径

先记住下面这条主线，后面遇到新名词时都把它放回这张图里。

```text
用户文本
→ tokenizer 把文本变成 token id
→ Runtime 创建一个请求对象
→ Scheduler 决定本轮处理哪些请求
→ KV Cache Manager 分配历史状态空间
→ ModelRunner 准备 tensor 并调用模型
→ Attention、GEMM、RMSNorm 等 kernel 在 GPU 上执行
→ 模型输出 logits，也就是下一个 token 的分数
→ Sampler 根据分数选择 token
→ 把新 token 加回请求
→ 继续下一轮，直到结束
→ token 转回文本并返回
```

这里有两个非常重要的阶段：

### Prefill

Prefill 是处理用户输入内容的阶段。

如果用户一次输入了 500 个 token，框架需要先处理这些已知内容，并为每层 Attention 建立 KV Cache。

Prefill 往往计算量较大，比较容易出现较大的 GEMM 和 Attention。

### Decode

Decode 是逐步生成新 token 的阶段。

普通自回归生成中，每轮通常只处理一个新 token，但需要读取之前积累的 KV Cache。

Decode 经常受到显存读取、小 kernel、CPU 发射 kernel、调度开销等因素影响。

现代框架可能会把长 Prefill 切成几段，也可能使用一次预测多个 token 的技术。因此“Prefill 一次处理完、Decode 永远一个 token”只是最初的简化理解。

---

## 3. 总周期和阶段安排

建议总周期约 7～8 个月。

```text
工作日：每天 1～2 小时
周末：一次连续 3～5 小时实验
每周：至少留下一个可以运行的脚本、一组结果或一页总结
```

核心路线约 28 周：

| 阶段 | 周期 | 项目 |
| --- | ---: | --- |
| 阶段 0 | 2 周 | 不使用 `generate()`，自己写最小生成循环 |
| 阶段 1 | 2 周 | 用三种方式跑通同一个小模型 |
| 阶段 2 | 3 周 | 给 Qwen3-0.6B 做一次“模型体检” |
| 阶段 3 | 4 周 | 给 nano-vLLM 安装“请求行车记录仪” |
| 阶段 4 | 4 周 | 做 KV Cache 与调度实验室 |
| 阶段 5 | 4 周 | 找出一个推理慢问题的真正原因 |
| 阶段 6 | 4 周 | 画出 nano-vLLM 到 vLLM 的对应地图 |
| 阶段 7 | 5 周 | 把一个 GPU kernel 完整接入推理框架 |

扩展路线约 6 周：

| 阶段 | 周期 | 项目 |
| --- | ---: | --- |
| 阶段 8 | 3 周 | 追踪量化权重从文件走到 kernel |
| 阶段 9 | 3 周 | 在两张 GPU 上实现最小 Tensor Parallel |

阶段 8 和阶段 9 的先后可以互换。考虑到你目前把多卡通信优先级放得较低，默认先学量化，再学双卡。

---

## 4. 阶段 0：不使用 `generate()`，自己写最小生成循环

建议周期：2 周。

### 4.1 为什么先做这个项目

很多教程一上来调用 `model.generate()`。它很方便，但把内部过程都藏起来了。

如果你连一次生成循环都没有亲手写过，后面看到 Prefill、Decode、logits、sampling 时容易只记住名词。

这个项目不训练模型，只使用已经训练好的小模型。

### 4.2 你在项目中扮演什么角色

你是一个“最小推理程序开发者”。

你的职责是：把文本变成 token，调用模型做 forward，从 logits 中选择下一个 token，再把它加入输入，循环生成。

### 4.3 需要理解的词

#### Shape

tensor 在每个维度上的大小。例如 `[1, 32]` 可以表示 batch 中有 1 条输入，每条输入有 32 个 token。

#### Dtype

tensor 中每个数字使用的数据格式，例如 FP32、FP16、BF16、INT8。它会影响精度、显存占用、访存量和硬件执行方式。

#### Device

tensor 当前存放和执行计算的设备，例如 CPU、`cuda:0` 或某种国产 GPU。模型权重和输入通常必须位于兼容的设备上。

#### Stride 和 contiguous

Stride 描述 tensor 沿每个维度移动一个位置时，底层内存地址要跨过多少元素。

`contiguous` 表示数据按照当前 shape 的常规顺序连续排放。转置等操作可能只改变 shape/stride 而不搬数据，因此 kernel 接入时不能只检查 shape。

#### `state_dict`

PyTorch 保存参数和 buffer 的“名称到 tensor”映射。阶段 2 会进一步学习它和模型权重文件的关系。

#### `nn.Module`

PyTorch 用来组织模型层和参数的基本对象。一个模型、一个 Attention 层或一个 Linear 层都可以是 `nn.Module`。

#### `forward`

模型的一次前向计算。输入 tensor 进入模型，输出新的 tensor。

#### `eval()`

告诉模型现在处于推理模式。它会关闭 Dropout 等只在训练时使用的随机行为。

#### `inference_mode()`

告诉 PyTorch 不需要记录梯度和反向传播所需的信息，从而减少额外开销。

#### `logits`

模型对词表中每个 token 给出的原始分数。分数越大，一般表示模型越倾向选择它。

#### Greedy sampling

每次直接选择分数最高的 token。这种方式没有随机性，适合最初验证流程。

### 4.3.1 训练知识只补到什么程度

这一阶段花半天到一天理解下面四点就够了：

1. 大语言模型训练时主要练习“根据前面的 token 预测下一个 token”。
2. Causal mask（因果遮罩）让当前位置不能偷看未来 token，否则训练和真实生成条件不一致。
3. 训练会根据错误修改权重；推理只读取已经训练好的权重。
4. 训练模式和推理模式可能有不同行为，因此推理前要调用 `eval()` 并关闭梯度记录。

这里不要求写反向传播，也不要求训练出一个模型。

### 4.4 你要做什么

1. 补齐推理所需的 PyTorch 基础：

```text
tensor 的 shape、dtype、device
tensor 的 stride 和 contiguous
nn.Module
state_dict
eval
inference_mode
CPU 与 GPU tensor 的移动
CUDA 同步与计时
```

2. 使用 Hugging Face 加载 Qwen3-0.6B 或同级别小模型。
3. 使用 tokenizer 把一句话变成 `input_ids`。
4. 直接调用一次 `model(**inputs)`，观察 logits 的 shape。
5. 取最后一个位置的 logits，用 `argmax` 选出下一个 token。
6. 把新 token 拼回输入，循环生成 10～20 个 token。
7. 和 `model.generate(do_sample=False)` 的结果做对比。
8. 打印每轮输入长度、logits shape、选中的 token id 和对应文字。

### 4.5 项目产物

```text
00_manual_decode.py
manual_decode_notes.md
```

笔记至少回答：

- 文本什么时候变成 token id？
- forward 的输入和输出是什么？
- 为什么只看最后一个位置的 logits？
- 新 token 为什么要加入下一轮输入？
- `generate()` 帮你隐藏了哪些工作？

### 4.6 完成标准

- 不使用 `generate()` 也能生成一小段文字。
- 能解释 tensor 的 device、dtype 和 shape 为什么必须正确。
- 能解释 `eval()` 和 `inference_mode()` 的作用。
- 能说清 logits 如何变成下一个 token。

### 4.7 做完后掌握的技能

- 能阅读最基本的 PyTorch 推理代码。
- 能理解模型 forward，而不是把模型当成完全黑盒。
- 为后面理解 Prefill、Decode 和 Sampler 建立基础。

### 4.8 暂时不要做什么

- 不学习完整反向传播。
- 不学习优化器。
- 不训练 Qwen。
- 不追求这份手写循环的性能。

---

## 5. 阶段 1：用三种方式跑通同一个小模型

建议周期：2 周。

可以继续使用：[[第01天-Qwen3-0.6B三种方式部署操作手册]]。

### 5.1 项目目标

使用同一个小模型分别跑通：

```text
Hugging Face Transformers
nano-vLLM
vLLM offline
可选：vLLM server
```

这里的目标是建立完整闭环，不是比较谁最快。

### 5.2 你在项目中扮演什么角色

你是一个“模型部署入门工程师”。

你的职责是准备环境、加载模型、发送输入、拿到输出、保存运行记录，并能解释三条调用路径的区别。

### 5.3 需要理解的词

#### Hugging Face Transformers

一个常用的模型和 tokenizer 库。它提供模型定义、权重加载和 `generate()` 等功能。

#### nano-vLLM

一个代码量较小的教学型推理 Runtime。它适合用来观察 Scheduler、KV Cache 和 ModelRunner。

#### vLLM

一个更接近生产环境的推理框架。它支持批处理、KV Cache 管理、多进程、在线服务和多种优化。

#### Offline inference

在 Python 程序里直接调用推理框架，程序运行完就退出，不启动长期服务。

#### Server

启动一个长期运行的服务进程，其他程序通过 HTTP 请求调用模型。

#### Chat template

聊天模板。它把 `system`、`user`、`assistant` 等结构化消息整理成模型训练时见过的文本格式。

同一句用户问题如果使用了不同 chat template，最终 token id 可能不同，因此输出和性能也可能不同。

#### Temperature 和 top-p

它们是控制生成随机性的参数。

Temperature 越低，模型越倾向选择高分 token；越高，选择会更分散。

Top-p 会从累计概率达到某个阈值的一小组候选 token 中采样。做严格对比时要固定这些参数。

#### Smoke test

“冒烟测试”，意思是先确认最基本的流程能运行。它不能证明性能好，也不能证明所有功能都正确。

### 5.4 你要做什么

1. 记录 GPU、驱动、Python、PyTorch 和框架版本。
2. 为每种框架使用独立环境，避免依赖冲突。
3. 把模型下载到固定路径。
4. 依次运行 HF、nano-vLLM、vLLM offline。
5. 如果时间允许，启动 vLLM server 并使用 `curl` 调用。
6. 保存每次运行的命令、日志和报错。
7. 记录输入 token 数、输出 token 数、总耗时和大致显存。
8. 画出三条外部调用路径。

### 5.5 关于性能数字的重要提醒

这一阶段记录的时间只用于确认程序能运行。

下面这个数字：

```text
总生成时间 / 输出 token 数
```

不是严格的 TPOT，因为总生成时间还包含 Prefill 和第一个 token。

严格性能测量会在阶段 5 学习。

### 5.6 项目产物

```text
01_hf_generate.py
02_nanovllm_generate.py
03_vllm_offline.py
day1_record.md
environment_versions.md
```

### 5.7 完成标准

- 三种 offline 路径至少成功两种，最终应尽量全部跑通。
- 能解释 tokenizer、模型和 Runtime 分别负责什么。
- 能解释 offline 和 server 的区别。
- 遇到依赖错误时，能记录“版本、现象、原因、解决办法”。

### 5.8 做完后掌握的技能

- 第一次完整部署并调用预训练模型。
- 理解模型目录、环境和推理代码之间的关系。
- 能使用 vLLM 的最基本入口。
- 建立以后阅读源码时所需的真实参照物。

---

## 6. 阶段 2：给 Qwen3-0.6B 做一次“模型体检”

建议周期：3 周。

### 6.1 项目目标

把一个模型目录从“看起来是一堆文件”变成一张可以理解的结构图。

项目名称可以叫：

```text
Qwen3-0.6B 模型加载与权重审计
```

### 6.2 你在项目中扮演什么角色

你是一个“模型加载审计工程师”。

你的职责不是训练模型，而是确认：

- 模型结构由哪些配置决定？
- 权重文件里有哪些 tensor？
- 每个权重放进代码中的哪个模块？
- 加载后模型是否完整、shape 是否正确？

### 6.3 需要理解的词

#### Config

模型的结构说明书，一般保存在 `config.json`。

它会记录层数、hidden size、head 数、KV head 数和 MLP 中间维度等信息。

#### Checkpoint

保存模型状态的文件集合。推理中最重要的是权重 tensor 和相关配置。

#### Safetensors

一种保存 tensor 的文件格式。可以读取 tensor 名称、shape 和 dtype，而不需要执行模型代码。

#### `state_dict`

PyTorch 中“参数名称到 tensor”的映射表。

例如：

```text
model.layers.0.self_attn.q_proj.weight
→ 一个具体权重 tensor
```

#### 权重映射

把 checkpoint 中的权重名称和模型代码中的 `nn.Module` 对应起来。

### 6.4 你要做什么

1. 打开并解释模型目录中的：

```text
config.json
tokenizer.json
tokenizer_config.json
generation_config.json
model.safetensors
model.safetensors.index.json（如果存在）
```

2. 写脚本打印 config 的关键字段。
3. 写脚本列出全部权重名、shape 和 dtype。
4. 把权重分类为：

```text
Embedding
Attention 的 Q/K/V/O
MLP 的 gate/up/down
RMSNorm
LM Head
```

5. 对照 Hugging Face 模型代码，找到每类权重的使用位置。
6. 使用 forward hook 或临时日志打印每个大模块的输入输出 shape。
7. 计算模型权重的理论显存：

```text
参数数量 × 每个参数的字节数
```

例如 BF16/FP16 通常每个参数约 2 字节。
8. 解释为什么实际显存还会包含临时 tensor、KV Cache、框架缓存等部分。

### 6.5 项目产物

```text
inspect_config.py
inspect_weights.py
trace_module_shapes.py
qwen_loading_audit.md
```

### 6.6 完成标准

- 给你一个陌生模型目录，知道先看哪些文件。
- 能根据 config 画出模型主要结构。
- 能从权重名定位到模型代码中的 module。
- 能解释 QKV 融合或拆分为什么会影响 loader。
- 能估算模型权重显存。

### 6.7 做完后掌握的技能

- 模型加载与权重检查。
- Hugging Face 模型代码阅读。
- shape mismatch 和权重缺失问题的初步定位。
- 为新模型适配和量化权重加载打基础。

---

## 7. 阶段 3：给 nano-vLLM 安装“请求行车记录仪”

建议周期：4 周。

### 7.1 项目目标

沿着一个请求，从 `LLM.generate()` 一直追踪到模型 forward 和输出 token。

不要一开始重写框架。先给关键位置增加结构化日志，就像给系统安装行车记录仪。

### 7.2 你在项目中扮演什么角色

你是一个“Runtime 调试工程师”。

你的职责是观察请求在各模块之间如何移动、状态如何变化，以及每轮真正送给 GPU 的数据是什么。

### 7.3 需要理解的词

#### Request / Sequence

框架内部用来表示一个生成任务的对象。它通常保存输入 token、已经生成的 token、当前状态和停止条件。

#### Scheduler

调度器。它决定本轮让哪些请求使用 GPU，以及每个请求处理多少 token。

#### ModelRunner

模型执行器。它把 Scheduler 选出的请求整理成 tensor，然后调用模型 forward。

#### Batch

一次一起送给 GPU 处理的一组请求或 token。

#### Sampler

采样器。它根据 logits、temperature、top-p 等规则选择下一个 token。

#### 状态机

一个对象会在若干状态之间变化。例如请求可能经历 waiting、running、finished。状态机就是描述这些状态和转换条件的方法。

### 7.4 你要做什么

1. 为 nano-vLLM 固定一个 commit，避免学习过程中源码突然变化。
2. 找到并阅读：

```text
外部 generate 入口
LLM Engine
Sequence
Scheduler
Block Manager
Model Runner
模型代码
Sampler
```

3. 为每个请求打印唯一 request id。
4. 在请求创建时打印 prompt 长度和最大生成长度。
5. 在每轮调度时打印 waiting、running、finished 数量。
6. 打印本轮是 Prefill 还是 Decode，以及本轮 token 数。
7. 在 ModelRunner 中打印输入 tensor 的 shape、dtype 和 device。
8. 打印 forward 前后时间。
9. 打印采样到的 token id。
10. 请求结束时打印结束原因和总生成长度。
11. 使用开关控制日志，避免永久污染正常运行路径。

### 7.5 项目产物

```text
runtime_trace.patch 或独立分支
runtime_trace_sample.log
runtime_flow.md
request_lifecycle.drawio / mermaid 图
```

### 7.6 完成标准

- 能不看源码讲清一个请求的生命周期。
- 能指出 Scheduler 和 ModelRunner 的职责边界。
- 能在日志中区分 Prefill 和 Decode。
- 能根据日志找到真正调用模型 forward 的位置。
- 能解释为什么推理 Runtime 不只是一个 `model(input_ids)` 循环。

### 7.7 做完后掌握的技能

- 阅读 Python 推理框架源码。
- 用日志验证调用关系和状态变化。
- 理解 Runtime 的最小模块划分。
- 为 KV Cache、调度和生产级 vLLM 学习打基础。

---

## 8. 阶段 4：做一个 KV Cache 与调度实验室

建议周期：4 周。

### 8.1 项目目标

通过可以重复的实验理解：为什么 Runtime 必须管理 KV Cache，以及 Scheduler 的选择怎样改变延迟、吞吐和显存。

### 8.2 你在项目中扮演什么角色

你是一个“推理资源调度实验工程师”。

你的职责是构造不同长度和不同并发的请求，观察显存块如何分配、请求如何排队，并修改一个简单策略验证结果。

### 8.3 需要理解的词

#### KV Cache

Attention 在处理历史 token 时产生的 Key 和 Value 缓存。

有了它，生成新 token 时不需要重新对整个历史上下文执行完整 forward。

#### Block

框架把 KV Cache 显存切成固定大小的小块，方便分配、回收和复用。它类似内存管理中的页。

#### Block table

记录“一个请求的逻辑 token 位置对应哪个物理 KV block”的映射表。

#### Continuous batching

连续批处理。不是等待一整批请求全部结束，而是在每一轮移除已完成请求并加入新请求。

#### Prefix cache

多个请求拥有相同开头时，复用已经计算好的前缀 KV Cache，减少重复计算。

#### 吞吐和延迟

吞吐表示单位时间完成多少工作；延迟表示单个请求等了多久。提高吞吐有时会让某些请求等待更久。

### 8.4 先学会一个 KV Cache 估算公式

对于普通 Attention，一个 token 的 KV Cache 大致与下面这些量成正比：

```text
层数
× KV head 数
× 每个 head 的维度
× K 和 V 两份
× 每个元素的字节数
```

GQA/MQA 的 KV head 更少，因此 KV Cache 通常比同规模 MHA 更省。

实际框架还会受到 block padding、对齐、并行切分和其他缓存影响。

### 8.5 你要做什么

1. 给 Block Manager 增加分配和释放日志。
2. 运行短 prompt，记录使用的 block 数。
3. 运行长 prompt，观察 block 数如何变化。
4. 改变最大生成长度，观察 Decode 过程中何时申请新 block。
5. 同时提交多个请求，观察 Continuous batching。
6. 构造一个长 prompt 和多个短 prompt，观察短请求是否被拖慢。
7. 打开或关闭 prefix cache，构造相同前缀请求并比较。
8. 修改一个简单策略，例如限制每轮最多处理的 Prefill token 数。
9. 对比修改前后的：

```text
短请求等待时间
长请求首 token 时间
每轮 batch token 数
总吞吐
KV block 使用量
```

### 8.6 项目产物

```text
kv_cache_formula.md
kv_block_trace.log
scheduler_experiments.py
scheduler_before_after.csv
kv_scheduler_report.md
```

### 8.7 完成标准

- 能解释 KV Cache 为什么存在以及它主要占用什么资源。
- 能画出 block table 到 KV Cache tensor 的关系。
- 能解释长 prompt 为什么会影响其他请求。
- 能解释 Scheduler 为什么是在延迟、吞吐、公平性和显存之间做取舍。
- 至少完成一次调度策略修改和前后对比。

### 8.8 做完后掌握的技能

- KV Cache 容量估算和生命周期分析。
- 请求调度实验设计。
- 从系统状态而不是单个 kernel 解释性能现象。

---

## 9. 阶段 5：找出一个推理慢问题的真正原因

建议周期：4 周。

### 9.1 项目目标

选择一个可以稳定复现的慢场景，使用计时、日志和性能工具找出证据，而不是凭感觉猜瓶颈。

### 9.2 你在项目中扮演什么角色

你是一个“端到端性能分析工程师”。

你的职责是先建立可靠基线，再把总耗时拆开，找到真正限制性能的部分，并通过最小修改验证判断。

### 9.3 需要理解的性能指标

#### TTFT

Time To First Token，从请求发出到收到第一个 token 的时间。

它包含排队、输入处理和 Prefill 等时间。

#### ITL

Inter-Token Latency，相邻两个输出 token 之间的时间。

它能反映输出是否平稳，也能看出某些轮次是否突然卡顿。

#### TPOT

Time Per Output Token。通常表示第一个 token 之后，平均生成一个输出 token 花费的时间。

#### E2E latency

End-to-End latency，从请求开始到完整答案结束的总时间。

#### Throughput

单位时间处理的 token 数或请求数，例如 output tokens/s、total tokens/s、requests/s。

#### P50 / P95 / P99

延迟的百分位数。例如 P99 表示 99% 的请求延迟不超过这个数。它比只看平均值更容易发现长尾慢请求。

#### Concurrency

并发度。表示同一时间有多少请求正在等待或处理。并发提高后，GPU 可能更忙，但排队和单请求延迟也可能增加。

#### Baseline

修改代码前的基准结果。没有 baseline，就无法证明修改是否有效。

#### Warmup

正式测量前先运行几次，让模型加载、编译、缓存和图捕获等一次性工作完成。

### 9.4 性能分析工具分别做什么

#### PyTorch Profiler

适合从 PyTorch 视角观察 CPU、GPU 操作和算子调用。

#### Nsight Systems

适合看整条时间线：CPU 什么时候提交 kernel、GPU 什么时候工作、哪里有空洞或同步。

#### Nsight Compute

适合深入分析单个 CUDA kernel，例如访存、occupancy 和指令效率。你已经比较熟悉这一层。

#### NVTX

在代码中添加带名字的时间区间。打开 Nsight Systems 时，可以看到“Scheduler”“Prepare Input”“Model Forward”等标签。

### 9.5 你要做什么

1. 固定模型、GPU、软件版本和采样参数。
2. 设计至少四类 workload：

```text
短输入 + 短输出
长输入 + 短输出
短输入 + 长输出
不同并发请求
```

这里的 workload 是测试工作负载，也就是输入长度、输出长度、请求数量和到达速度的组合。

3. 每组实验先 warmup，再重复多次。
4. 同时记录 TTFT、ITL、TPOT、E2E、吞吐和显存。
5. 使用流式 server 测量真实的首 token 和 token 间隔。
6. 在 Runtime 关键位置添加 NVTX 标记。
7. 用 Nsight Systems 判断主要问题属于：

```text
CPU 调度慢
CPU 发射 kernel 太碎
GPU 中间有空洞
Attention 慢
GEMM 慢
KV Cache 读取慢
同步过多
通信等待
```

8. 对比 eager 和图执行。

`eager` 指每次由程序正常逐个发起操作；图执行指先记录一段相对固定的 GPU 工作，以后重复重放，减少 CPU 发射开销。

9. 如果条件允许，对比 `torch.compile`。

`torch.compile` 会尝试分析计算图、融合操作并生成更高效的执行方式，但可能有编译时间和动态 shape 限制。

10. 选择一个最可能的瓶颈，做一次最小修改并重新测量。

### 9.6 项目产物

```text
benchmark_config.yaml
benchmark_results.json/csv
nsys_trace
profile_case_study.md
```

报告必须包含：

```text
问题是什么
如何稳定复现
修改前数据
时间线证据
你的判断
做了什么修改
修改后数据
是否有副作用
结论适用于哪些条件
```

### 9.7 完成标准

- 不再用单次运行的 tokens/s 判断性能。
- 能正确区分 TTFT、ITL、TPOT 和总耗时。
- 能从时间线上区分 CPU 空洞、GPU kernel 和同步等待。
- 能用实验支持“瓶颈在哪里”的判断。

### 9.8 做完后掌握的技能

- 科学的端到端 benchmark。
- 从 Runtime 一直追到 kernel 的性能定位。
- CUDA Graph 和编译优化的适用条件判断。

---

## 10. 阶段 6：画出 nano-vLLM 到 vLLM 的对应地图

建议周期：4 周。

### 10.1 项目目标

把已经理解的 nano-vLLM 模块映射到生产级 vLLM，理解生产框架为什么多出进程、抽象和错误处理。

不要以“读完整个 vLLM”为目标。

### 10.2 你在项目中扮演什么角色

你是一个“生产推理框架架构调查员”。

你的职责是沿着固定主线找对应模块，并解释每一层额外复杂度解决了什么问题。

### 10.3 需要理解的词

#### Entrypoint

程序入口。offline API、HTTP server 都是不同入口。

#### Engine Core

推理引擎的核心进程或核心逻辑，负责调度和请求状态推进。

#### Worker

真正控制一个或多个设备、执行模型的工作进程。

#### Executor

组织 Worker 执行任务的组件，可能处理单进程、多进程或多机执行。

#### IPC

Inter-Process Communication，进程间通信。不同进程不能直接随意访问彼此的普通变量，需要通过队列、共享内存、socket 等方式交换数据。

#### Platform

框架对一种硬件平台的抽象。它告诉 vLLM 当前是什么设备、支持哪些 dtype、使用哪个 Attention backend、如何查询显存等。

#### Attention Backend

框架用来执行 Attention 的一套接口和具体实现。不同硬件可以选择不同实现。

### 10.4 你要做什么

1. 为 vLLM 固定版本或 commit。
2. 分别找到 offline 和 server 入口。
3. 画出 API Server、Engine Core、Worker/ModelRunner 的进程关系。
4. 沿一个请求追踪：

```text
入口
→ Engine
→ Scheduler
→ KV Cache Manager
→ Worker
→ ModelRunner
→ Attention/CustomOp
```

5. 对照 nano-vLLM，写出模块映射表。
6. 找到当前平台选择逻辑。
7. 找到 Attention backend 的选择和调用位置。
8. 找到框架中的 CustomOp 或可替换层接口。
9. 找到性能指标如何从 Engine 传到服务层。
10. 做一个很小的修改，例如增加一个内部计时或调试 metric。

### 10.5 Serving 在这一阶段学到什么程度

需要理解：

```text
HTTP 请求如何进入
为什么要流式返回
请求在哪里排队
服务层和 Engine 为什么分进程
怎么产生 TTFT/TPOT 等指标
进程崩溃时错误在哪一层出现
```

暂时不需要深入：

```text
Kubernetes
复杂网关
跨地域容灾
大规模自动扩缩容
```

### 10.6 项目产物

```text
nano_to_vllm_mapping.md
vllm_process_architecture.md
vllm_request_trace.md
一个小型代码修改
```

### 10.7 完成标准

- 能解释 vLLM 为什么比 nano-vLLM 多出多个进程和组件。
- 能区分 API Server、Engine、Worker 和 ModelRunner。
- 能找到 Platform、Attention backend 和 CustomOp 的入口。
- 遇到问题时知道从哪个进程、哪类日志开始查。

### 10.8 做完后掌握的技能

- 阅读生产级推理框架。
- 理解多进程推理架构。
- 找到新硬件和新 kernel 的主要接入点。

---

## 11. 阶段 7：把一个 GPU kernel 完整接入推理框架

建议周期：5 周。

这是核心路线中最重要的综合项目。

### 11.1 项目目标

不要只写一个 benchmark 很快的 kernel，而是完成下面整条链路：

```text
PyTorch reference
→ GPU kernel
→ C++/Python 接口
→ 算子注册
→ 框架 dispatch
→ 模型层调用
→ 正确性测试
→ 端到端 benchmark
→ profile 证明实际调用
```

第一版建议选择 RMSNorm 或 SiLU-and-mul 这类边界清晰的算子，而不是直接从 Attention 开始。

原因是 Attention 还牵涉 Prefill/Decode、KV Cache layout、block table 和多种 backend。先用简单算子学会完整接入方法，再迁移到 Attention，会更稳。

### 11.2 你在项目中扮演什么角色

你是一个“推理后端适配工程师”。

你的职责不只是提供 kernel，还要对接口、分发、兼容性、正确性、回退路径和端到端收益负责。

### 11.3 需要理解的词

#### Custom Op

不属于 PyTorch 默认算子集合、由开发者自己注册的算子。

#### Dispatcher

PyTorch 或框架根据设备、dtype、shape 等条件选择具体实现的机制。

#### Dispatch

执行“选择哪个实现”的过程。

#### Fallback

当新硬件或新 kernel 不支持某种 shape/dtype 时，退回到一个较慢但正确的通用实现。

#### FakeTensor / Meta 实现

只根据输入的 shape、dtype、device 推导输出信息，不真正计算数据。编译器可利用它分析计算图。

#### ABI

二进制接口兼容性。PyTorch、编译器和扩展版本不匹配时，可能出现 undefined symbol 等错误。

#### Capability matrix

能力矩阵。用表格记录某种硬件或后端支持什么、不支持什么、限制是什么。

### 11.4 项目分为两部分

#### A. 在租用的 NVIDIA GPU 上练习公开链路

1. 写 PyTorch reference。
2. 使用你熟悉的方法实现或选择一个 CUDA kernel。
3. 通过 PyTorch Custom Op 注册。
4. 为编译路径提供必要的 shape/dtype 推导。
5. 写覆盖不同 shape、dtype、非连续 tensor 和边界值的测试。
6. 在 nano-vLLM 或 vLLM 模型层增加 dispatch。
7. 不满足条件时回退到原实现。
8. 用日志和 profiler 证明新 kernel 被真实调用。
9. 对比单算子性能和端到端性能。

#### B. 映射到国产 GPU/新硬件适配

在不记录公司机密的前提下，为目标硬件建立通用能力矩阵：

```text
设备发现
device guard
stream / event
内存分配器
pinned host memory
dtype
随机数
算子注册
通用 fallback
图执行
编译器
Attention backend
KV Cache layout
多进程
通信库
Profiler
打包、版本和 CI
```

对于每一项回答：

```text
是否支持？
对外接口是什么？
与 CUDA 语义有什么差异？
vLLM 在哪里依赖它？
如何做最小正确性测试？
如何做性能测试？
```

能力矩阵中的常见词可以这样理解：

| 名词 | 通俗解释 |
| --- | --- |
| device guard | 管理当前线程正在使用哪张设备，并在临时切换后恢复原设备 |
| stream | 设备上的异步工作队列；不同 stream 中的任务可能重叠执行 |
| event | 插入 stream 的时间点标记，可用于同步或设备侧计时 |
| allocator | 显存管理员，负责申请、缓存、复用和释放设备内存 |
| pinned host memory | 不会被操作系统随意换出的主机内存，通常有利于异步 CPU/设备传输 |
| dtype | FP16、BF16、INT8 等数据格式以及硬件对它们的支持情况 |
| 随机数 | sampling 等随机操作需要可重复、可保存状态的设备随机数生成器 |
| 图执行 | 记录一段设备工作并重复执行，以减少 CPU 反复提交操作的开销 |
| 编译器接入 | 让图编译器认识新设备和自定义算子，并能正确推导 shape、dtype 等信息 |
| 通信库 | 多设备之间执行 all-reduce、all-gather 等操作的底层库 |
| Profiler | 采集 CPU、设备、算子、内存和通信时间线的性能分析工具 |
| CI | 持续集成；代码变化后自动执行构建、正确性测试和必要的性能回归检查 |

### 11.5 正确性验证必须逐层进行

```text
单算子输出对齐
→ 一个模型层对齐
→ 多层 hidden state 对齐
→ 最终 logits 对齐
→ greedy token 序列对齐
→ 小评测集检查
→ 长时间和边界 shape 稳定性
```

其中 hidden state 是模型中间层输出的 tensor。

只看“模型回答看起来合理”不能证明后端正确。

### 11.6 项目产物

```text
custom_op 源码和构建脚本
reference 实现
correctness_tests
kernel_benchmark
framework_integration.patch
end_to_end_benchmark
profile_trace
hardware_capability_matrix.md
kernel_to_runtime_report.md
```

### 11.7 完成标准

- 新 kernel 能通过框架真实调用，而不是只在独立 benchmark 中运行。
- 不支持的输入有明确报错或 fallback。
- 有单算子、模型层和端到端三层正确性证据。
- 有单算子和端到端两层性能数据。
- 能解释为什么 kernel 变快但端到端可能没有明显变化。
- 能说清新硬件接入 vLLM 时需要补齐哪些平台能力。

### 11.8 做完后掌握的技能

- 从 kernel 到推理 Runtime 的完整集成。
- PyTorch 与框架的算子分发。
- 新硬件能力梳理。
- 正确性、兼容性、性能和工程可维护性的综合判断。

---

## 12. 阶段 8：追踪量化权重从文件走到 kernel

建议周期：3 周。

### 12.1 项目目标

选择一种框架已经支持的量化格式，追踪量化权重、scale 和其他信息如何从 checkpoint 进入量化 kernel。

### 12.2 你在项目中扮演什么角色

你是一个“量化推理集成工程师”。

你不只是优化 INT8/FP8 kernel，还要理解量化模型怎么保存、加载、分发和验证。

### 12.3 需要理解的词

#### 量化

用更低位数的数据表示权重、激活或 KV Cache，以减少显存、访存或计算开销。

#### Scale

低精度整数和原始浮点数之间的缩放系数。

#### Zero point

某些整数格式用来表示数值零的位置偏移。

#### Per-channel / Per-group

不同通道或不同小组分别使用 scale。组越小通常越容易保持精度，但额外元数据和计算也更多。

#### Weight-only quantization

只量化权重，激活仍使用 FP16/BF16 等格式。

#### KV Cache quantization

降低 KV Cache 的存储精度，重点解决长上下文和高并发时的显存、带宽问题。

### 12.4 你要做什么

1. 选择一个已经被框架支持的量化小模型。
2. 打印量化 checkpoint 中的权重名、shape、dtype 和 scale。
3. 找到 loader 如何识别量化配置。
4. 找到普通 Linear 如何被替换成量化 Linear。
5. 找到权重 pack 或 reorder 的位置。
6. 找到最终 kernel dispatch。
7. 判断反量化发生在 kernel 内还是 kernel 外。
8. 对比非量化和量化模型的：

```text
权重显存
总显存
TTFT
TPOT
吞吐
输出差异
```

9. 使用固定数据集做最小精度检查，而不是只看一个问题的回答。

### 12.5 项目产物

```text
quantized_weight_inspector.py
quantized_weight_to_kernel.md
accuracy_sanity_check.md
quantization_benchmark.csv
```

### 12.6 完成标准

- 能解释量化为什么不保证一定更快。
- 能找到 scale、权重重排和 kernel dispatch。
- 能区分权重量化和 KV Cache 量化解决的问题。
- 能同时报告性能收益和精度代价。

---

## 13. 阶段 9：在两张 GPU 上实现最小 Tensor Parallel

建议周期：3 周。

### 13.1 项目目标

租用两张 NVIDIA GPU，从最小通信实验开始，实现 Column Parallel Linear 和 Row Parallel Linear，再观察小模型双卡推理。

### 13.2 你在项目中扮演什么角色

你是一个“多设备推理实验工程师”。

你的职责是保证多进程能够启动、tensor 正确切分、通信结果正确，并解释为什么双卡不一定比单卡快。

### 13.3 需要理解的词

#### Rank

分布式程序中一个进程的编号。

#### World size

参与这一组计算的总进程数或设备数。

#### Process group

一组可以彼此通信的进程。

#### Collective communication

多个设备共同参加的通信操作，例如 all-reduce、all-gather、broadcast。

#### All-reduce

每张卡提供一份数据，把它们求和或做其他归约后，让每张卡都得到最终结果。

#### All-gather

每张卡提供一小段数据，最后让各卡都拿到拼接后的完整数据。

#### Tensor Parallel

把一个大矩阵或一层模型切到多张设备上计算，再通过通信合并结果。

### 13.4 你要做什么

1. 启动两个进程，打印各自 rank 和设备。
2. 写最小 all-reduce 实验。
3. 写最小 all-gather 实验。
4. 实现 Column Parallel Linear，并和单卡结果对齐。
5. 实现 Row Parallel Linear，并和单卡结果对齐。
6. 找到 QKV、O projection、MLP 在 TP 中如何切分。
7. 用小模型运行 TP=2。
8. 使用时间线观察计算和通信。
9. 对比：

```text
单卡耗时
双卡耗时
每卡显存
通信耗时
计算耗时
```

### 13.5 对结果的正确预期

小模型在双卡上可能比单卡更慢，这是正常且有学习价值的结果。

原因可能是：

```text
每张卡计算量太小
通信时间超过省下的计算时间
进程同步开销
GPU 之间连接较慢
batch 太小
```

这个项目的第一目标是理解切分和通信，不是证明双卡一定加速。

### 13.6 项目产物

```text
distributed_hello.py
all_reduce_demo.py
all_gather_demo.py
column_parallel_linear.py
row_parallel_linear.py
tp_correctness_test.py
tp_profile_report.md
```

### 13.7 完成标准

- 两卡结果和单卡 reference 在允许误差内一致。
- 能解释每个通信点为什么出现。
- 能从 profile 中找到通信操作。
- 能解释双卡变慢或变快的原因。

---

## 14. 小模型和租赁 GPU 怎么安排

### 日常学习

Qwen3-0.6B 或同级别模型足够完成：

```text
流程跑通
模型加载
源码阅读
Runtime trace
KV Cache 和 Scheduler 实验
正确性测试
框架接口修改
```

### 阶段性性能测试

如果预算允许，可以短时租用能运行 3B～8B 模型的 NVIDIA GPU。

原因是 0.6B 模型太小，CPU 调度和 kernel launch 开销可能占比很高，不能代表更大模型中的 GEMM、显存带宽和通信特征。

### 双卡测试

只在阶段 9 集中租双卡即可。

记录 GPU 型号和卡间连接方式。不同型号、不同连接方式的结果不能直接横向比较。

---

## 15. 每周怎么学

推荐固定节奏：

### 工作日一：读概念

- 只读本周项目直接需要的概念。
- 每个概念用自己的话写两三句。
- 写出它的输入、输出和影响的指标。

### 工作日二：读源码

- 只沿一条调用路径读。
- 记录函数被谁调用、又调用谁。
- 不追求一次读懂整个仓库。

### 工作日三：加日志或做小实验

- 修改一个很小的点。
- 确认修改真的运行到了。

### 工作日四：整理问题

- 区分“我不知道这个词”和“我知道词但不知道代码在哪里”。
- 把未解决问题放进问题清单，不要无限扩展当前主题。

### 周末：完成一次完整实验

```text
写清假设
→ 固定环境和输入
→ 运行 baseline
→ 做一个修改
→ 重复测量
→ 保存结果
→ 写结论
```

---

## 16. 每次实验都使用同一个模板

```markdown
# 实验名称

## 我想回答的问题

## 我的判断/假设

## 环境
- GPU：
- 驱动：
- Python：
- PyTorch：
- 框架版本/commit：
- 模型：

## 输入条件
- prompt token 数：
- output token 数：
- batch/concurrency：
- dtype：
- 采样参数：

## Baseline

## 修改内容

## 修改后结果

## Profile 或日志证据

## 结论

## 结论只适用于哪些条件

## 仍未解决的问题
```

---

## 17. 怎么判断可以进入下一阶段

每个阶段结束时检查五件事：

1. 我能不能不用照着原文，讲清这个项目解决了什么问题？
2. 我能不能画出这个阶段的输入、处理过程和输出？
3. 我能不能在源码中找到关键入口？
4. 我有没有实际修改代码或设计实验？
5. 我的结论有没有日志、正确性测试或性能数据支持？

如果其中三项做不到，就不要急着进入下一阶段。

“看过”和“掌握”不同：

```text
看过：我认识这个词
理解：我知道它为什么存在
会使用：我能运行和配置它
会调试：我能找到问题发生的位置
掌握：我能修改它并用实验证明结果
```

---

## 18. 完成核心路线后，你应该具备什么能力

完成阶段 0～7 后，目标不是“精通所有推理框架”，而是能够：

- 独立把一个预训练模型跑起来。
- 解释文本如何一步步变成输出 token。
- 阅读模型 config、权重和 forward。
- 沿着请求追踪 Scheduler、KV Cache、ModelRunner 和 kernel。
- 正确测量 TTFT、TPOT、吞吐和显存。
- 使用时间线判断瓶颈属于 CPU、GPU、算子、同步还是框架调度。
- 把一个新 kernel 接进推理框架，而不是只做独立算子 benchmark。
- 为一种新硬件建立 Runtime 能力矩阵。
- 知道 PyTorch、vLLM 和硬件 Runtime 之间有哪些适配边界。
- 用正确性和性能证据向框架、算子或硬件同事解释结论。

完成量化和双卡扩展后，还应能够：

- 追踪量化权重从 checkpoint 到 kernel 的完整路径。
- 理解基本 Tensor Parallel 切分和通信。
- 初步判断量化或多卡为什么可能没有端到端收益。

---

## 19. 当前不需要优先学习的内容

下面这些以后可能有价值，但暂时不要打断主线：

```text
完整模型训练
复杂优化器和大规模训练框架
Kubernetes 和复杂微服务治理
一次读完全部 vLLM 源码
所有量化论文
多机大规模部署
MoE、MTP、多模态
Prefill/Decode 分离
复杂投机解码
完整 MLIR/TVM 编译器体系
```

等阶段 7 完成后，再根据真实工作需要选择其中一项扩展。

---

## 20. 和已有文档的关系

- [[LLM推理基础概念]]：作为概念地图使用，不需要背诵。
- [[第01天-Qwen3-0.6B三种方式部署操作手册]]：用于阶段 1 跑通环境。
- [[学习计划]]：保留为较专业、较完整的原始版本。
- [[nano-vllm 学习总结]]：目前只有外部链接，阶段 3 完成后应改写成自己的源码和实验总结。

这份通俗版是主执行计划。以后每完成一个阶段，就在对应章节下面增加：

```text
实际开始/结束日期
使用的代码版本
项目链接
实验结果
遇到的问题
是否达到完成标准
```
