# infer_opt

按一份固定学习计划，用一连串可运行的小项目，逐步学会大模型部署和推理优化。

完整计划见 [`plan/LLM推理优化工程师学习计划-通俗版.md`](plan/LLM推理优化工程师学习计划-通俗版.md)。本仓库是计划的执行现场：每个阶段对应一个目录，留下脚本、笔记和实验结果。

## 学什么、不学什么

目标不是训练模型，也不是 Kubernetes 运维。最终要形成的能力是：

```text
看到推理慢或跑失败
→ 判断问题属于哪一层（加载 / 调度 / 显存 / 算子 / 通信 / 服务）
→ 沿请求路径找到对应代码
→ 设计最小实验验证
→ 改代码
→ 证明结果仍正确，且端到端有收益
```

主线从「一次 forward 产出一个 token」走到 Prefill / Decode、KV Cache、调度、性能定位，再接到 nano-vLLM 和 vLLM，最后把一个 GPU kernel 完整接入推理框架。

## 一次请求的路径

后面每个阶段都在这条链上往下拆一层：

```text
文本 → tokenizer → Runtime 建请求 → Scheduler
     → KV Cache → ModelRunner / 模型 forward
     → Attention / GEMM / RMSNorm 等 kernel
     → logits → Sampler → 新 token → 循环直到结束
```

两个阶段要先分清：

- **Prefill**：处理整段输入，建立 KV Cache，计算量较大。
- **Decode**：逐步生成新 token，通常一次一个，更容易被显存和调度开销卡住。

## 仓库结构

```text
infer_opt/
  plan/           学习计划（路线图，先读这个）
  stage0/         手写最小生成循环
  stage1/         三种方式跑通同一个小模型
  stage2/ …       后续阶段，做到哪一层加哪一层
  local/          本机大文件，不进 git（环境、权重、whl）
```

`local/` 只跟踪一份说明，见 [`local/guide.md`](local/guide.md)。虚拟环境、模型权重和体积过大的安装包都放在那里，不会提交到仓库。

## 学习路线

核心约 28 周，扩展约 6 周。每个阶段都是一个能跑完、能讲清楚的小项目。

| 阶段 | 状态 | 项目 |
| --- | --- | --- |
| 0 | 进行中 | 不使用 `generate()`，自己写最小生成循环 |
| 1 | 已有部署脚本 | 用 HF / nano-vLLM / vLLM 跑通 Qwen3-0.6B |
| 2 | 未开始 | 给 Qwen3-0.6B 做一次「模型体检」 |
| 3 | 未开始 | 给 nano-vLLM 安装「请求行车记录仪」 |
| 4 | 未开始 | KV Cache 与调度实验室 |
| 5 | 未开始 | 找出一个推理慢问题的真正原因 |
| 6 | 未开始 | 画出 nano-vLLM 到 vLLM 的对应地图 |
| 7 | 未开始 | 把一个 GPU kernel 完整接入推理框架 |
| 8 | 未开始 | 追踪量化权重从文件走到 kernel |
| 9 | 未开始 | 在两张 GPU 上实现最小 Tensor Parallel |

阶段 0 的入口：[`stage0/guide.md`](stage0/guide.md)。

## 怎么跑

模型权重和虚拟环境不在仓库里。需要一块 NVIDIA GPU、Python 3.10+，并把 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 放到 `local/models/Qwen3-0.6B`。

```bash
hf download Qwen/Qwen3-0.6B --local-dir local/models/Qwen3-0.6B
source local/venvs/.venv-hf/bin/activate   # 环境需自行创建
cd stage0
python 00a_tensor_basics.py
```

脚本要在对应阶段目录里运行。例如 `stage0` 依赖同目录的 `common.py`。

## 约定

- 计划在 `plan/`，代码和笔记在 `stageN/`，大文件在 `local/`。
- 只有仓库根目录使用 `README.md`。子目录里的说明文档叫 `guide.md`。
- 每周至少留下一个可运行脚本、一组结果或一页总结。
- 性能结论必须绑定测量条件：短上下文上成立的优化，长上下文上未必成立。
- 两个都正确的实现，在 bf16 下输出可以不完全一样；对齐时要比 token 路径，不要只比文字。
