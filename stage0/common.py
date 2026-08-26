"""
阶段 0 公共工具：模型路径、加载、prompt 构造、计时。

把这些"每个脚本都要重复一遍"的样板代码集中放在这里，
让 00a~00e 每个脚本只保留它真正想让你看到的那部分逻辑。

这个文件本身不需要直接运行。
"""

import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 基本配置
# ---------------------------------------------------------------------------

# 仓库根目录下的 local/models/，不写本机绝对路径
_REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(_REPO_ROOT / "local" / "models" / "Qwen3-0.6B")# Qwen3-0.6B 模型路径 

# 计算设备。有 N 卡就用 GPU，否则退回 CPU（0.6B 在 CPU 上也能跑，只是慢）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 权重的数据格式。
#   bfloat16：16 位浮点，显存占用是 float32 的一半，数值范围和 float32 一样宽，
#            Ada 架构（RTX 40 系）原生支持，是当前推理的默认选择。
# CPU 上对 bf16 支持差，所以退回 float32。
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
print(f"DTYPE: {DTYPE}")


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(verbose: bool = True):
    """加载 tokenizer 和模型，返回 (tokenizer, model)。

    这里出现的每一步，都是 `pipeline()` 或 `LLM()` 帮你隐藏掉的工作。
    """
    # tokenizer 负责「文本 <-> token id」的双向转换。
    # 它本身不含神经网络权重，只是一套分词规则加一张词表。
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # from_pretrained 做了三件事：读 config.json 搭出网络结构、
    # 从 model.safetensors 读权重、把权重按名字填进网络。
    # 注意参数名是 dtype（transformers v5）；v4 时代叫 torch_dtype，现在已弃用。
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=DTYPE)

    # from_pretrained 默认把模型放在 CPU 上，要自己搬到 GPU。
    model = model.to(DEVICE)

    # eval() 关闭 Dropout / BatchNorm 这类"训练时才有的随机行为"。
    # Qwen3 推理时其实没有 Dropout，但养成习惯：推理前一定 eval()。
    model.eval()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[load] 模型类型   : {type(model).__name__}")
        print(f"[load] 参数量     : {n_params / 1e9:.3f} B")
        print(f"[load] 权重 dtype : {next(model.parameters()).dtype}")
        print(f"[load] 权重 device: {next(model.parameters()).device}")
        print(f"[load] 词表大小   : {model.config.vocab_size}")

    return tokenizer, model


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------

# 阶段 0 全程用这一句话，方便横向对比不同脚本的结果
DEFAULT_QUESTION = "用一句话说明什么是大语言模型推理。"


def build_prompt(tokenizer, question: str = DEFAULT_QUESTION) -> str:
    """把用户的一句话，套上 Qwen3 的对话模板，变成模型真正看到的文本。

    为什么不直接把问题原文喂给模型？
    因为 Qwen3-0.6B 是"指令模型"，它在训练时看到的输入长这样：
        <|im_start|>user\n问题<|im_end|>\n<|im_start|>assistant\n
    只有按这个格式喂，它才知道"现在轮到我回答了，答完要收尾"。
    直接喂裸问题，它会退化成续写机器，一直复读停不下来。

    enable_thinking=False：关掉 Qwen3 的思考链模式。
    开启时模型会先输出一大段 <think>...</think> 推理过程，动辄几百 token，
    不利于阶段 0 观察循环。等阶段 3 做请求追踪时再打开它。
    """
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,            # 只要文本，不要直接转成 id，方便我们打印出来看
        add_generation_prompt=True,  # 末尾补上 "<|im_start|>assistant\n"，提示模型开始答
        enable_thinking=False,
    )


def get_eos_ids(model) -> set:
    """取出所有表示"生成结束"的 token id，统一成集合。

    坑点：Qwen3 的 generation_config.json 里 eos_token_id 是一个【列表】
        [151645, 151643]   分别是 <|im_end|> 和 <|endoftext|>
    不是单个整数。手写循环时如果拿它去和单个 token id 做 ==，永远不相等，
    循环就停不下来。00d 会专门演示这个坑。
    """
    eos = model.generation_config.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return set(eos)


# ---------------------------------------------------------------------------
# 计时
# ---------------------------------------------------------------------------

def cuda_sync():
    """等待 GPU 上所有已下发的计算真正执行完。

    CUDA 是异步的：Python 里 model(...) 返回时，GPU 可能还在算。
    不同步就计时，测到的只是"把任务丢给 GPU 花了多久"，不是"算完花了多久"。
    00a 最后一节会用数字把这个差距演示出来。
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """配合 with 使用的计时器，进出时都会同步 GPU。

    用法:
        with Timer() as t:
            ...要计时的代码...
        print(t.elapsed)
    """

    def __enter__(self):
        cuda_sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        cuda_sync()
        self.elapsed = time.perf_counter() - self._t0
        return False


def banner(title: str):
    """打印分节标题，让长输出好读一些。"""
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)
