"""02 Prefill 和 Decode 是两种不同的活。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: F401

from cached_loop import kv_cache_bytes_per_token


def run(model, records):
    print("##### 2. Prefill 和 Decode 是两种不同的活 ###")
    prefill = records[0]
    decodes = records[1:]
    decode_times = sorted(r["elapsed_ms"] for r in decodes)
    med_decode = decode_times[len(decode_times) // 2] if decode_times else 0.0

    print(f"""
  Prefill : 喂入 {prefill['fed']} 个 token，一次算完，耗时 {prefill['elapsed_ms']:.2f} ms
  Decode  : 每次只喂 1 个 token，共 {len(decodes)} 次，中位数 {med_decode:.2f} ms

一个反直觉的地方：prefill 处理了 {prefill['fed']} 个 token，
decode 每次只处理 1 个，但两者耗时差不多。

为什么？因为这两步都要把模型的全部 1.1GB 权重从显存里读一遍。
读权重的时间跟你喂多少 token 无关。
prefill 摊到每个 token 上非常划算，decode 则完全是在"为了 1 个 token 读 1.1GB"。

这就是 Decode 阶段被称为"访存受限"(memory-bound) 的原因，
也是为什么 batch 化对 decode 收益极大——
反正都要读这 1.1GB，不如一次多带几个请求一起算。
（阶段 4 的 continuous batching 就是在做这件事。）
""")

    per_token = kv_cache_bytes_per_token(model)
    cfg = model.config
    print("  缓存的代价（KV Cache 不是免费的）:")
    print(
        f"    每个 token 占 2 x {cfg.num_hidden_layers}层 x "
        f"{cfg.num_key_value_heads}个kv头 x {cfg.head_dim}维 x 2字节 "
        f"= {per_token/1024:.0f} KB"
    )
    print(
        f"    本次生成 {records[-1]['cache_len']} 个 token，"
        f"缓存约 {per_token*records[-1]['cache_len']/1024**2:.1f} MB"
    )
    print(
        f"    如果是 8192 长上下文，一条请求就要 "
        f"{per_token*8192/1024**2:.0f} MB"
    )
    print(
        f"    16GB 显卡去掉 1.1GB 权重，最多同时装下约 "
        f"{int((15*1024**3) / (per_token*8192))} 条这样的请求"
    )
    print()
    print("  显存换时间，就是 KV Cache 的本质。阶段 4 做的 PagedAttention，")
    print("  解决的正是「这块显存怎么分配才不浪费」的问题。")

if __name__ == "__main__":
    from importlib import import_module

    _tok, model, _ids, records = import_module("01_cached_loop").run()
    run(model, records)
