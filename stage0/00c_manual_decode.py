"""
00c：手写生成循环（阶段 0 的核心产物）。

不调用 model.generate()，自己把下面这个循环写出来：

    while 还没到上限:
        1. 把当前整条序列喂给模型，做一次 forward
        2. 取最后一个位置的 logits
        3. argmax 选出下一个 token
        4. 把它拼到序列末尾
        5. 如果它是结束符，跳出

这一版【故意不使用 KV Cache】：每一步都把整条序列重新算一遍。
这么写效率很差，但有两个好处：
    - 逻辑最直白，没有任何隐藏状态
    - 你能亲眼看到耗时随序列变长而上升，这就是 KV Cache 存在的理由

运行:
    ../local/venvs/.venv-hf/bin/python 00c_manual_decode.py
"""

import torch

from common import (
    DEFAULT_QUESTION,
    Timer,
    banner,
    build_prompt,
    get_eos_ids,
    load_model_and_tokenizer,
)

# 最多生成多少个新 token。设成 64 是因为这个问题模型大约 30 步就会自己结束，
# 留一倍余量既能看到正常收尾，又不会等太久。
MAX_NEW_TOKENS = 64


def manual_greedy_decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    verbose: bool = True,
):
    """不使用 generate()，手写 greedy 解码循环。

    返回 (generated_ids, records)
        generated_ids : list[int]  只包含新生成的 token id，不含 prompt
        records       : list[dict] 每一步的观测数据，供后面画耗时曲线
    """
    eos_ids = get_eos_ids(model)

    # 把 prompt 变成 id，并搬到模型所在设备上
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = input_ids.shape[1]

    generated_ids = []
    records = []

    if verbose:
        print(f"prompt 长度: {prompt_len} tokens")
        print(f"结束符 id  : {sorted(eos_ids)} "
              f"-> {[tokenizer.decode([i]) for i in sorted(eos_ids)]}")
        print(f"最多生成   : {max_new_tokens} tokens")
        print()
        print(f"{'步':>3} {'输入长度':>6} {'logits shape':>22} "
              f"{'token id':>9} {'耗时ms':>8}  文本")
        print("-" * 78)

    with torch.inference_mode():
        for step in range(max_new_tokens):
            # ---- 第 1 步：整条序列做一次 forward ----------------------------
            # 注意这里每轮喂的都是完整的 input_ids，长度在不断变长。
            # 前面那些 token 上一轮已经算过一模一样的结果了，这里全部重算，
            # 纯属浪费——这正是 00e 要解决的问题。
            with Timer() as t:
                outputs = model(input_ids=input_ids)

            logits = outputs.logits

            # ---- 第 2 步：只取最后一个位置 ----------------------------------
            # [batch, seq_len, vocab] -> [vocab]
            # 为什么是最后一个位置：见 00b 第 3 节
            next_token_logits = logits[0, -1, :]

            # ---- 第 3 步：greedy 选词 --------------------------------------
            # argmax 就是"选分数最高的那个"，没有任何随机性。
            next_id = next_token_logits.argmax(dim=-1).item()

            # ---- 第 4 步：拼回输入 ------------------------------------------
            # 下一轮的输入 = 这一轮的输入 + 刚生成的这个 token。
            # 不拼回去的话，模型下一轮看到的上下文和这一轮完全一样，
            # 就会永远输出同一个词。
            next_token = torch.tensor([[next_id]], device=input_ids.device)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            generated_ids.append(next_id)
            records.append({
                "step": step,
                "input_len": logits.shape[1],
                "logits_shape": tuple(logits.shape),
                "token_id": next_id,
                "elapsed_ms": t.elapsed * 1000,
            })

            if verbose:
                piece = tokenizer.decode([next_id])
                print(f"{step:>3} {logits.shape[1]:>6} "
                      f"{str(tuple(logits.shape)):>22} "
                      f"{next_id:>9} {t.elapsed*1000:>8.2f}  {piece!r}")

            # ---- 第 5 步：判断要不要停 --------------------------------------
            # 用 in 判断是否属于结束符集合，不能用 == 和单个值比。
            # 原因见 common.get_eos_ids 的注释，00d 会专门演示写错的后果。
            if next_id in eos_ids:
                if verbose:
                    print("-" * 78)
                    print(f"遇到结束符，第 {step} 步正常收尾")
                break
        else:
            if verbose:
                print("-" * 78)
                print(f"达到 {max_new_tokens} 上限，被强制截断（模型还想继续说）")

    return generated_ids, records


def main():
    banner("0. 准备")
    tokenizer, model = load_model_and_tokenizer()

    prompt = build_prompt(tokenizer, DEFAULT_QUESTION)
    print()
    print(f"问题: {DEFAULT_QUESTION}")

    # 预热：第一次 forward 有几百毫秒的 CUDA 初始化开销，
    # 不预热的话第 0 步的耗时会特别大，干扰后面看趋势。
    with torch.inference_mode():
        warm = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        model(input_ids=warm)

    banner("1. 手写生成循环（不使用 KV Cache）")
    generated_ids, records = manual_greedy_decode(model, tokenizer, prompt)

    banner("2. 生成结果")
    # skip_special_tokens=True 会把 <|im_end|> 之类的标记去掉，只留可读文本
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(text)

    print()
    print("带特殊标记的原始版本（能看到结尾的 <|im_end|>）:")
    print(f"  {tokenizer.decode(generated_ids, skip_special_tokens=False)!r}")

    banner("3. 每步耗时：先别急着下结论")

    n = len(records)
    times = [r["elapsed_ms"] for r in records]
    print(f"总共 {n} 步，累计 {sum(times)/1000:.2f} 秒，"
          f"平均每 token {sum(times)/n:.2f} ms")

    if n >= 10:
        head = sum(times[:5]) / 5
        tail = sum(times[-5:]) / 5
        print()
        print(f"  前 5 步平均: {head:6.2f} ms  (序列长约 {records[2]['input_len']})")
        print(f"  后 5 步平均: {tail:6.2f} ms  (序列长约 {records[-3]['input_len']})")
        print(f"  变化: {(tail/head - 1)*100:+.1f}%")

    print()
    print("  每步耗时（每个 # 约 1ms）:")
    step_show = max(1, n // 20)
    for r in records[::step_show]:
        bar = "#" * int(r["elapsed_ms"])
        print(f"    step {r['step']:>3} len={r['input_len']:>3} "
              f"{r['elapsed_ms']:>6.2f}ms {bar}")

    print("""
按常理推断，序列越长算得越多、应该越慢。但上面这组数字基本是平的，
甚至可能还略微下降，中间偶尔冒出几个尖峰（那多半是桌面程序在抢 GPU）。

这不是测错了，而是【测量尺度不对】。原因有两个：

  1. 序列才 20~50 个 token，Attention 那部分计算量小到可以忽略。
  2. 每跑一次 forward，要往 GPU 下发几百个 kernel（28 层，每层好几个）。
     这个固定开销加起来十几毫秒，把真正的计算时间盖住了。

也就是说，此刻的瓶颈不是"算得多"，而是"启动得多"和"权重读得多"。
这个模型只有 0.6B，每步都要把全部 1.1GB 权重从显存读一遍——
这件事的耗时和序列长度无关，却是当前的主要开销。

想看到增长趋势，得把序列拉长到固定开销盖不住的程度。下面直接测。
""")

    banner("4. 换个尺度：forward 耗时 vs 序列长度")

    print("用随机 token 构造不同长度的输入，各测 10 次取平均：")
    print()
    print(f"  {'序列长度':>8} {'forward耗时':>12} {'每token':>10}   相对 16 长度")
    print("  " + "-" * 56)

    sweep = [16, 64, 256, 512, 1024, 2048, 4096]
    base_ms = None
    with torch.inference_mode():
        for length in sweep:
            fake = torch.randint(1000, 50000, (1, length), device=model.device)
            for _ in range(3):                      # 预热
                model(input_ids=fake, logits_to_keep=1)
            with Timer() as t:
                for _ in range(10):
                    model(input_ids=fake, logits_to_keep=1)
            ms = t.elapsed * 1000 / 10
            base_ms = base_ms or ms
            print(f"  {length:>8} {ms:>10.2f} ms {ms/length:>9.4f} "
                  f"     {ms/base_ms:>5.2f}x")

    print("""
现在趋势就清楚了：

  - 512 以内，耗时几乎是一条水平线。固定开销主导，跟序列长度没关系。
  - 1024 往上开始明显爬升。
  - 4096 相比 2048，长度翻倍但耗时涨了两倍多——这是超线性。
    因为 Attention 要让每个 token 和前面所有 token 算相关性，
    计算量按 O(n^2) 增长，长上下文时它会变成主角。

顺带一个很实用的教训：性能结论必须绑定测量条件。
"KV Cache 能加速"这句话在 4096 上下文是对的，在 50 个 token 上却是错的。
00d 会用数字把这句话验证一遍。
""")

    banner("5. 这一版浪费在哪")

    total_tokens_forwarded = sum(r["input_len"] for r in records)
    print(f"""
这 {n} 步一共把 {total_tokens_forwarded:,} 个 token 喂进了模型。
但真正"新"的 token 只有 {n} 个，其余 {total_tokens_forwarded - n:,} 个都是
在把上一轮已经算过的内容原样重算一遍。

重复计算占比约 {(1 - n/total_tokens_forwarded)*100:.1f}%。

注意区分两件事：
  - 【浪费的计算量】确实是 {(1 - n/total_tokens_forwarded)*100:.0f}%，这是客观事实。
  - 【浪费的时间】在当前这么短的序列上几乎看不出来，因为瓶颈在别处。

每个 token 在 Attention 里算出的 Key 和 Value，只取决于它自己和它前面的内容，
后面每一轮都完全一样。既然一样，就没必要反复算——存下来复用即可。
这就是 KV Cache。

下一步: 00d_kv_cache.py 把它实现出来，并回答"到底什么时候才划算"。
""")

    banner("动手改改看")
    print("""
1. 把第 4 步"拼回输入"那两行注释掉，看看会输出什么。
   （提示：模型每轮看到的上下文都一样）

2. 把 argmax 换成随机抽样:
       probs = torch.softmax(next_token_logits.float(), dim=-1)
       next_id = torch.multinomial(probs, num_samples=1).item()
   跑三次，结果还一样吗？这就是 do_sample=True 在做的事。

3. 把 MAX_NEW_TOKENS 改成 5，观察最后打印的是"正常收尾"还是"强制截断"。

4. 把 logits[0, -1, :] 改成 logits[0, 0, :]（取第一个位置），
   生成的文字会变成什么样？为什么？

5. 第 4 节的扫描里把 logits_to_keep=1 去掉，重新跑。
   4096 那一行的耗时变化大吗？（提示：lm_head 要输出 4096x151936 个数）
""")


if __name__ == "__main__":
    main()
