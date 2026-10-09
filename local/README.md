# local：不进 git 的本机文件

这个目录存放体积过大、无法推送到 GitHub 的本机产物。
除了本 README，目录内其它内容都被 `.gitignore` 忽略。

```text
local/
  venvs/          Python 虚拟环境
    .venv-hf                 Hugging Face / 阶段 0
    .venv-nano310-fa274      nano-vLLM
    .venv-vllm-cu12          vLLM
  models/         模型权重
    Qwen3-0.6B
  wheels/         体积过大的安装包（如 flash-attn 的 wheel）
```

三套环境的 Python、PyTorch 和框架版本，以及各自对应哪条脚本，见 [`stage1/environment_versions.md`](../stage1/environment_versions.md)。

## 激活环境

```bash
source local/venvs/.venv-hf/bin/activate
```

## 下载模型

```bash
hf download Qwen/Qwen3-0.6B --local-dir local/models/Qwen3-0.6B
```
