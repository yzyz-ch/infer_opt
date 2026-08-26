"""
00e：和 model.generate() 对齐，验证手写循环写对了没有。

这是阶段 0 的最后一关，也是最有价值的一关。
"能跑出通顺的句子"不等于"写对了"——必须和官方实现逐 token 比对才算数。

这个脚本会做三件事：
    1. 把「无缓存手写」「有缓存手写」「generate()」三者的输出摆在一起比
    2. 深入分析为什么其中一个对得上、另一个对不上
    3. 演示停止条件写错会发生什么

运行:
    ../local/venvs/.venv-hf/bin/python 00e_compare_with_generate.py
"""

import importlib

import torch
from transformers import DynamicCache

from common import (
    DEFAULT_QUESTION,
    banner,
    build_prompt,
    get_eos_ids,
    load_model_and_tokenizer,
)

# 文件名以数字开头，不能写 import 00c_manual_decode，
# 但 importlib.import_module 接受任意字符串名字，可以绕过这个语法限制。
manual_mod = importlib.import_module("00c_manual_decode")
cache_mod = importlib.import_module("00d_kv_cache")

MAX_NEW_TOKENS = 64


def show_tokens(tokenizer, ids, limit=12):
    """把 token id 列表变成一行紧凑的可读形式，方便肉眼对比。"""
    pieces = [tokenizer.decode([i]) for i in ids[:limit]]
    tail = " ..." if len(ids) > limit else ""
    return " | ".join(p.replace("\n", "\\n") for p in pieces) + tail


def main():
    banner("0. 准备")
    tokenizer, model = load_model_and_tokenizer()
    prompt = build_prompt(tokenizer, DEFAULT_QUESTION)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    eos_ids = get_eos_ids(model)

    print(f"\n问题: {DEFAULT_QUESTION}")
    print(f"prompt 长度: {prompt_ids.shape[1]} tokens")

    # ---------------------------------------------------------------------
    banner("1. 三种方式各跑一遍")
    # ---------------------------------------------------------------------

    print("(1) 无缓存手写循环 —— 00c 的实现")
    ids_nocache, _ = manual_mod.manual_greedy_decode(
        model, tokenizer, prompt, MAX_NEW_TOKENS, verbose=False)

    print("(2) 有缓存手写循环 —— 00d 的实现")
    ids_cache, _ = cache_mod.cached_greedy_decode(
        model, tokenizer, prompt, MAX_NEW_TOKENS, verbose=False)

    print("(3) 官方 model.generate(do_sample=False)")
    with torch.inference_mode():
        out = model.generate(
            prompt_ids,
            # 显式传 attention_mask。不传的话 transformers 会警告：
            # Qwen3 的 pad_token 和 eos_token 是同一个 id，它无法自动推断
            # 哪些位置是真实内容、哪些是填充。这里没有填充，全 1 即可。
            attention_mask=torch.ones_like(prompt_ids),
            do_sample=False,        # 关掉随机采样，改用 greedy，才能和 argmax 对齐
            max_new_tokens=MAX_NEW_TOKENS,
        )
    # generate 返回的是「prompt + 新生成」，要把 prompt 部分切掉
    ids_generate = out[0, prompt_ids.shape[1]:].tolist()

    print()
    print("注：如果上面出现关于 temperature / top_p / top_k 的警告，那是正常的。")
    print("    generation_config.json 里默认写了 do_sample=true 和这些采样参数，")
    print("    我们传了 do_sample=False，于是那些参数用不上，transformers 就提醒一句。")

    # ---------------------------------------------------------------------
    banner("2. 三者对比")
    # ---------------------------------------------------------------------

    def first_diff(a, b):
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return i
        return None if len(a) == len(b) else min(len(a), len(b))

    rows = [
        ("无缓存手写", ids_nocache),
        ("有缓存手写", ids_cache),
        ("generate()", ids_generate),
    ]
    print(f"  {'实现':<12} {'token数':>7}  文本")
    print("  " + "-" * 74)
    for name, ids in rows:
        print(f"  {name:<12} {len(ids):>7}  "
              f"{tokenizer.decode(ids, skip_special_tokens=True)}")

    print()
    d_nc = first_diff(ids_nocache, ids_generate)
    d_c = first_diff(ids_cache, ids_generate)
    print(f"  无缓存手写 vs generate(): "
          f"{'完全一致' if d_nc is None else f'第 {d_nc} 步开始分歧'}")
    print(f"  有缓存手写 vs generate(): "
          f"{'完全一致' if d_c is None else f'第 {d_c} 步开始分歧'}")

    print("""
这个结果第一眼很反直觉：明明"无缓存"那版逻辑最朴素、最不容易写错，
反而对不上；"有缓存"那版复杂得多，却完全一致。

原因不在逻辑，在数值路径。
generate() 内部默认是开 KV Cache 的，所以它和"有缓存手写"走的是同一条数值路径。
逻辑正确 + 数值路径相同 = 逐 token 一致。

而"无缓存"那版逻辑同样正确，只是数值路径不同，于是结果允许有细微差别。
""")

    # ---------------------------------------------------------------------
    banner("3. 把分歧的那一步拆开看")
    # ---------------------------------------------------------------------

    if d_nc is None:
        print("这次没有出现分歧（有一定偶然性），跳过本节。")
        print("多换几个问题试试，很容易复现。")
    else:
        k = d_nc
        print(f"第 {k} 步之前两边完全相同，就从这一步开始查。")
        print(f"  共同前缀（最后几个词）: "
              f"...{show_tokens(tokenizer, ids_nocache[max(0,k-5):k])}")
        print()

        # 构造出分歧那一步的输入：prompt + 前 k 个已生成的 token
        prefix = torch.cat([
            prompt_ids,
            torch.tensor([ids_nocache[:k]], device=model.device, dtype=prompt_ids.dtype),
        ], dim=1) if k > 0 else prompt_ids

        with torch.inference_mode():
            # 路径 A：整条序列一次前向（00c 的算法）
            logits_a = model(input_ids=prefix).logits[0, -1, :].float()

            # 路径 B：先 prefill 前面全部，再单独 decode 最后一个 token（00d 的算法）
            cache = DynamicCache()
            model(input_ids=prefix[:, :-1], past_key_values=cache, use_cache=True)
            logits_b = model(input_ids=prefix[:, -1:], past_key_values=cache,
                             use_cache=True).logits[0, -1, :].float()

        top_a = torch.topk(logits_a, 3)
        top_b = torch.topk(logits_b, 3)

        print("  两条数值路径给出的前 3 名:")
        print(f"    {'排名':>4}  {'整条前向(00c)':<28} {'缓存解码(00d)':<28}")
        for r in range(3):
            ta = f"{tokenizer.decode([top_a.indices[r].item()])!r} {top_a.values[r].item():.4f}"
            tb = f"{tokenizer.decode([top_b.indices[r].item()])!r} {top_b.values[r].item():.4f}"
            print(f"    {r+1:>4}  {ta:<28} {tb:<28}")

        margin_a = (top_a.values[0] - top_a.values[1]).item()
        diff = (logits_a - logits_b).abs().max().item()
        print()
        print(f"  两条路径 logits 的最大差异: {diff:.4f}")
        print(f"  第一名和第二名的分数差距  : {margin_a:.4f}")
        print()
        if margin_a < diff:
            print("  差距 < 数值误差，所以谁排第一完全由浮点舍入决定，翻转是必然的。")
        else:
            print("  这一步差距虽然大于误差，但前面某处的微小差异已经改变了上下文。")

        print("""
  结论：bf16 只有 8 位尾数，不同矩阵形状会走不同 kernel，累加顺序不同，
  logits 就会有 0.0x 量级的抖动。平时无所谓，但只要某一步的前两名咬得很紧，
  这点抖动就足以让 argmax 翻转。而生成是自回归的，一步走岔，后面全岔。

  这不是 bug，是 bf16 推理的固有特性。
  真实项目里做正确性验证时，判据通常是这样定的：
      - 逐 token 完全一致（要求两边数值路径相同，最严格）
      - top-1 一致率 / 困惑度差异在阈值内（路径不同时用这个）
      - 人工评测输出质量（最终兜底）
  拿"文字一模一样"当唯一标准，会冤枉很多其实正确的实现。
""")

    # ---------------------------------------------------------------------
    banner("4. 停止条件写错会怎样")
    # ---------------------------------------------------------------------

    print("先看看这个模型的结束符到底是怎么配置的：")
    print(f"  config.json          里 eos_token_id = {model.config.eos_token_id}")
    print(f"  generation_config    里 eos_token_id = "
          f"{model.generation_config.eos_token_id}")
    print(f"  tokenizer            里 eos_token_id = {tokenizer.eos_token_id}")
    print()
    print("  三处不一样！generation_config 里是个【列表】，另外两处是单个整数。")
    print("  真正生效的是 generation_config，因为 generate() 用的就是它。")
    print()
    print(f"  列表里两个 id 分别是: "
          f"{[(i, tokenizer.decode([i])) for i in model.generation_config.eos_token_id]}")

    print()
    print("现在故意写错停止条件，看看后果：")
    print("  错误写法: if next_id == model.generation_config.eos_token_id")
    print("  这是拿一个 int 去和一个 list 比，永远返回 False。")
    print()

    wrong_stop = model.generation_config.eos_token_id   # 一个列表
    ids = prompt_ids.clone()
    buggy_out = []
    stop_should_have_been = None
    with torch.inference_mode():
        for step in range(48):
            logits = model(input_ids=ids, logits_to_keep=1).logits
            nid = logits[0, -1, :].argmax(dim=-1).item()
            buggy_out.append(nid)
            ids = torch.cat([ids, torch.tensor([[nid]], device=ids.device)], dim=1)
            if nid in eos_ids and stop_should_have_been is None:
                stop_should_have_been = step
            if nid == wrong_stop:     # 永远不成立
                break

    print(f"  本该在第 {stop_should_have_been} 步停下，实际跑满了 {len(buggy_out)} 步。")
    print()
    print("  越过结束符之后模型输出了什么:")
    print(f"    {tokenizer.decode(buggy_out, skip_special_tokens=False)!r}")
    print()
    print("""  可以看到它吐出 <|im_end|> 之后并没有安静下来，而是自顾自开启了新一轮对话，
  甚至开始扮演用户提问。因为对模型来说 <|im_end|> 只是又一个普通 token，
  "到此为止"这件事从来不是模型决定的，而是【外面这层循环】决定的。

  这就是 generate() 替你做、而你很容易忽略的一件事。
  正确写法是判断是否属于结束符集合:
      if next_id in eos_ids:   # eos_ids = set(generation_config.eos_token_id)
          break
""")

    # ---------------------------------------------------------------------
    banner("5. generate() 到底替你做了什么")
    # ---------------------------------------------------------------------

    print("""
把阶段 0 走一遍之后，这个问题就有了具体答案：

  1. 循环控制       —— 反复 forward，直到满足停止条件
  2. 取最后一个位置 —— 从 [batch, seq, vocab] 里挑出 [:, -1, :]
  3. 选词策略       —— greedy / temperature / top-k / top-p / beam search
  4. 拼接上下文     —— 把新 token 接回序列
  5. 停止条件       —— EOS 集合、max_new_tokens、stop strings
  6. KV Cache 管理  —— 建缓存、prefill、decode 时只喂一个 token
  7. attention_mask 和 position_ids 的维护
  8. 批处理与 padding —— 一批请求长度不一时怎么对齐
  9. 采样参数的读取与校验（generation_config）

阶段 0 亲手写了 1~6，这已经足够让你在后面读 nano-vLLM 和 vLLM 源码时，
认出它们各自对应哪个模块，而不是看到一堆陌生名词。

7~9 留到后面的阶段再补。
""")

    banner("阶段 0 完成标准自查")
    print("""
对照下面几条，能不看代码讲清楚就算过关：

  [ ] 文本是在哪一步变成 token id 的？之后模型还看得到文字吗？
  [ ] forward 的输入和输出分别是什么形状？各维度什么含义？
  [ ] 为什么只取最后一个位置的 logits？前面那些为什么用不上？
  [ ] 新 token 为什么必须拼回输入？不拼会怎样？
  [ ] eval() 和 inference_mode() 各自关掉了什么？
  [ ] KV Cache 存的是什么？为什么短上下文时它不省时间？
  [ ] Prefill 和 Decode 的计算特征有什么不同？
  [ ] 为什么两个都正确的实现，生成的文字可以不一样？

答不上来的那条，回去重跑对应的脚本。
""")


if __name__ == "__main__":
    main()
