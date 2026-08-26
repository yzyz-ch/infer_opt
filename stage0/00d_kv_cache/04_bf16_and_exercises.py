"""04 结果会和 00c 不一样（bf16 数值路径）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: F401

from common import banner


def run():
    banner("4. 一个必须知道的坑：结果会和 00c 不一样")
    print("""
把 00c 和 00d 的输出放一起看，你会发现两段话不完全相同。

数学上，用不用缓存应该得到一模一样的结果。实际上不会，原因是 bf16 精度：

  - 00c 每步做的是 [50, 1024] 这种形状的矩阵乘法
  - 00d 每步做的是 [1, 1024] 这种形状的矩阵乘法

形状不同，cuBLAS 会挑不同的 kernel，累加顺序也不同。
bf16 只有 8 位尾数，这点顺序差异就足以让 logits 差个 0.05 左右。

平时这无所谓。但当某一步的第一名和第二名分数非常接近时，
这 0.05 就足以让 argmax 翻转，选出不同的词。
而生成是自回归的——选错一个词，后面整段话都会顺着新方向走。

所以：
  - 不要用"两段文字一样"来验证实现是否正确
  - 要验证，就得让数值路径也一致

00e 会把这件事彻底讲清楚，并给出正确的验证方法。
""")

    banner("动手改改看")
    print("""
1. 把 decode 阶段的 input_ids=cur 改成喂整条序列，同时保留 cache。
   会报错还是结果错？想想为什么。

2. 在 benchmark 里把 n_new 从 32 改成 128，加速比变大还是变小？为什么？

3. 把 DynamicCache() 挪到 for 循环里面（每步新建一个空缓存），
   生成的文字会变成什么样？

4. 算一算：如果模型是 7B（32层、32个kv头、head_dim 128），
   8192 上下文的 KV Cache 要占多少显存？
""")


if __name__ == "__main__":
    run()
