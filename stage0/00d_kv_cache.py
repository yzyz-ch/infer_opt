"""
00d：给手写循环装上 KV Cache。

00c 里我们发现：每一步都把整条序列重算一遍，97% 的计算是重复的。
这个脚本把重复的部分缓存下来，并回答一个比"能不能加速"更重要的问题——
【到底在什么条件下才真的加速】。

先说结论，免得你被网上的说法误导：
    在 50 个 token 的短对话里，KV Cache 几乎不省时间，甚至可能略慢。
    在 4096 token 的长上下文里，它能快 4 倍以上。
这个脚本会用数字把这条分界线量出来。

运行:
    ../local/venvs/.venv-hf/bin/python 00d_kv_cache.py
"""

import torch
from transformers import DynamicCache

from common import (
    DEFAULT_QUESTION,
    Timer,
    banner,
    build_prompt,
    get_eos_ids,
    load_model_and_tokenizer,
)

MAX_NEW_TOKENS = 64


def cached_greedy_decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    verbose: bool = True,
):
    """使用 KV Cache 的 greedy 解码。

    和 00c 的区别只有一处，但很关键：
        00c：每轮把【整条序列】喂进去
        00d：第一轮喂整条 prompt，之后每轮只喂【刚生成的那一个 token】

    之所以只喂一个也能算对，是因为前面所有 token 的 Key/Value 都存在 cache 里，
    Attention 可以直接从缓存取，不需要重新过一遍网络。

    返回 (generated_ids, records)
    """
    eos_ids = get_eos_ids(model)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # DynamicCache 是 transformers v5 的缓存容器，会随着生成自动变长。
    # 注意：v4 时代 past_key_values 是一个嵌套 tuple，网上老教程都是那个写法，
    # 在 v5 上直接用 tuple 会报错。这里必须用 Cache 对象。
    cache = DynamicCache()

    generated_ids = []
    records = []

    if verbose:
        print(f"{'步':>3} {'本轮喂入':>8} {'缓存长度':>8} {'阶段':>8} "
              f"{'耗时ms':>8}  文本")
        print("-" * 74)

    with torch.inference_mode():
        # ---- Prefill：把整条 prompt 一次性算完，填满缓存 --------------------
        # 这一步和 00c 的第 0 步做的事完全一样，区别是把 K/V 留了下来。
        # 推理框架里这个阶段就叫 Prefill（预填充）。
        with Timer() as t:
            outputs = model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,   # 只要最后一个位置的 logits，见 00b 第 3 节
            )
        next_id = outputs.logits[0, -1, :].argmax(dim=-1).item()
        generated_ids.append(next_id)
        records.append({
            "step": 0, "fed": input_ids.shape[1], "cache_len": cache.get_seq_length(),
            "phase": "prefill", "elapsed_ms": t.elapsed * 1000, "token_id": next_id,
        })
        if verbose:
            print(f"{0:>3} {input_ids.shape[1]:>8} {cache.get_seq_length():>8} "
                  f"{'prefill':>8} {t.elapsed*1000:>8.2f}  "
                  f"{tokenizer.decode([next_id])!r}")

        # ---- Decode：之后每轮只喂一个新 token ------------------------------
        # 推理框架里这个阶段叫 Decode（解码）。
        # Prefill 和 Decode 的计算特征完全不同，这是后面所有调度优化的基础：
        #   Prefill 一次处理很多 token，计算密集，容易把 GPU 算力吃满
        #   Decode 每次只处理一个 token，访存密集，GPU 算力大量闲置
        if next_id not in eos_ids:
            for step in range(1, max_new_tokens):
                cur = torch.tensor([[next_id]], device=input_ids.device)

                with Timer() as t:
                    outputs = model(
                        input_ids=cur,          # 注意：只有 1 个 token
                        past_key_values=cache,  # 历史信息全在这里
                        use_cache=True,
                        logits_to_keep=1,
                    )

                next_id = outputs.logits[0, -1, :].argmax(dim=-1).item()
                generated_ids.append(next_id)
                records.append({
                    "step": step, "fed": 1, "cache_len": cache.get_seq_length(),
                    "phase": "decode", "elapsed_ms": t.elapsed * 1000,
                    "token_id": next_id,
                })

                if verbose:
                    print(f"{step:>3} {1:>8} {cache.get_seq_length():>8} "
                          f"{'decode':>8} {t.elapsed*1000:>8.2f}  "
                          f"{tokenizer.decode([next_id])!r}")

                if next_id in eos_ids:
                    break

    if verbose:
        print("-" * 74)

    return generated_ids, records


def benchmark(model, context_len: int, n_new: int = 32):
    """在指定的上下文长度下，对比"重算全部"和"用缓存"两种方式的解码速度。

    用随机 token 造一个指定长度的假上下文。内容是乱的没关系，
    我们只关心耗时，不关心它生成什么。
    """
    base = torch.randint(1000, 50000, (1, context_len), device=model.device)

    # --- 方式一：00c 的做法，每步重算整条序列 ---
    def run_no_cache():
        ids = base.clone()
        with torch.inference_mode():
            for _ in range(n_new):
                logits = model(input_ids=ids, logits_to_keep=1).logits
                nid = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, nid], dim=1)

    # --- 方式二：00d 的做法，prefill 一次 + 每步只喂一个 token ---
    def run_cache():
        cache = DynamicCache()
        with torch.inference_mode():
            out = model(input_ids=base, past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
            nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(n_new - 1):
                out = model(input_ids=nid, past_key_values=cache,
                            use_cache=True, logits_to_keep=1)
                nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    run_no_cache()   # 预热
    run_cache()

    with Timer() as t1:
        run_no_cache()
    with Timer() as t2:
        run_cache()

    return t1.elapsed, t2.elapsed


def kv_cache_bytes_per_token(model) -> int:
    """算一个 token 的 K/V 要占多少字节。

    公式: 2(K和V) x 层数 x kv头数 x 每头维度 x 每个数的字节数
    这里 kv 头数是 8 而注意力头数是 16，因为 Qwen3 用了 GQA
    （多个查询头共享一组 K/V），本身就是一种省缓存的设计。
    """
    cfg = model.config
    return (2 * cfg.num_hidden_layers * cfg.num_key_value_heads
            * cfg.head_dim * torch.finfo(model.dtype).bits // 8)


def main():
    banner("0. 准备")
    tokenizer, model = load_model_and_tokenizer()
    prompt = build_prompt(tokenizer, DEFAULT_QUESTION)

    # 预热。注意要把 prefill 和 decode 两种形状都跑一遍：
    # GPU 会为每种矩阵形状单独挑选 kernel，第一次遇到新形状要额外花时间，
    # 只热了 prefill 的话，第一个 decode 步会出现一个刺眼的尖峰。
    with torch.inference_mode():
        warm = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        warm_cache = DynamicCache()
        model(input_ids=warm, past_key_values=warm_cache, use_cache=True)
        one = torch.tensor([[warm[0, -1].item()]], device=model.device)
        model(input_ids=one, past_key_values=warm_cache, use_cache=True)

    banner("1. 带 KV Cache 的生成循环")
    print(f"问题: {DEFAULT_QUESTION}")
    print()
    generated_ids, records = cached_greedy_decode(model, tokenizer, prompt)

    print()
    print("生成结果:")
    print(f"  {tokenizer.decode(generated_ids, skip_special_tokens=True)}")

    banner("2. Prefill 和 Decode 是两种不同的活")

    prefill = records[0]
    decodes = records[1:]
    # 用中位数而不是平均数：GPU 上偶发的尖峰（别的程序抢资源）会把平均值拉高，
    # 中位数更能代表"典型的一步要多久"。做性能分析时这是个好习惯。
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
    print(f"  缓存的代价（KV Cache 不是免费的）:")
    print(f"    每个 token 占 2 x {cfg.num_hidden_layers}层 x "
          f"{cfg.num_key_value_heads}个kv头 x {cfg.head_dim}维 x 2字节 "
          f"= {per_token/1024:.0f} KB")
    print(f"    本次生成 {records[-1]['cache_len']} 个 token，"
          f"缓存约 {per_token*records[-1]['cache_len']/1024**2:.1f} MB")
    print(f"    如果是 8192 长上下文，一条请求就要 "
          f"{per_token*8192/1024**2:.0f} MB")
    print(f"    16GB 显卡去掉 1.1GB 权重，最多同时装下约 "
          f"{int((15*1024**3) / (per_token*8192))} 条这样的请求")
    print()
    print("  显存换时间，就是 KV Cache 的本质。阶段 4 做的 PagedAttention，")
    print("  解决的正是「这块显存怎么分配才不浪费」的问题。")

    banner("3. 关键问题：什么时候才真的加速")

    print(f"在不同上下文长度下，各生成 32 个 token，对比两种方式：")
    print()
    print(f"  {'上下文长度':>10} {'重算全部':>12} {'用缓存':>12} {'加速比':>8}")
    print("  " + "-" * 48)

    for ctx in (128, 512, 1024, 2048, 4096):
        t_no, t_yes = benchmark(model, ctx, n_new=32)
        print(f"  {ctx:>10} {t_no:>10.2f} s {t_yes:>10.2f} s "
              f"{t_no/t_yes:>7.2f}x")

    print("""
读这张表要注意：

  - 上下文很短时，加速比接近 1，甚至可能小于 1（缓存反而略慢）。
    因为此时瓶颈是 kernel 启动和读权重，而 KV Cache 一点也帮不上这两件事。
    它还额外带来了每层维护缓存的 Python 开销，所以可能倒亏一点。

  - 上下文越长，加速比越大。因为"重算全部"的代价随长度线性甚至超线性增长，
    而"用缓存"的每步代价基本不变。

真实服务场景的上下文动辄几千上万 token，所以 KV Cache 是必备的。
但你要记住它成立的前提，而不是把"KV Cache 更快"当成一条无条件的教条。

这也是这个阶段最该带走的思维方式：
任何优化都有生效范围，先量出范围，再决定用不用。
""")

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
    main()
