"""01 加载模型。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: F401

import torch

from common import load_model_and_tokenizer


def run():
    print("##### 0. 加载模型 ###")
    tokenizer, model = load_model_and_tokenizer()
    if torch.cuda.is_available():
        print(
            f"[load] 加载后显存占用: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )
        print("       约等于 0.6B 参数 x 2 字节(bf16)，和预期吻合")
    return tokenizer, model

if __name__ == "__main__":
    run()
