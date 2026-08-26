"""
00b：只做一次 forward，不循环。

这个脚本的唯一目的，是把"模型 forward 的输入和输出到底是什么"看清楚。
只要这一步想明白了，00c 的生成循环就只是把它重复 N 遍而已。

要看到的四件事：
    1. 文本是在哪一步变成 token id 的
    2. forward 吃进去什么、吐出来什么
    3. logits 有 seq_len 行，但我们只用得上最后一行
    4. 从 logits 到"下一个词"中间发生了什么

运行:
    ../local/venvs/.venv-hf/bin/python 00b_one_forward.py
"""

import torch

from common import (
    DEFAULT_QUESTION,
    Timer,
    banner,
    build_prompt,
    load_model_and_tokenizer,
)


banner("0. 加载模型")
tokenizer, model = load_model_and_tokenizer()

if torch.cuda.is_available():
    print(f"[load] 加载后显存占用: "
          f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print("       约等于 0.6B 参数 x 2 字节(bf16)，和预期吻合")


# ---------------------------------------------------------------------------
banner("1. 文本 -> token id")
# ---------------------------------------------------------------------------

print(f"你的问题原文:\n  {DEFAULT_QUESTION}")

prompt = build_prompt(tokenizer, DEFAULT_QUESTION)
print()
print("套上对话模板之后，模型真正看到的文本:")
print(f"  {prompt!r}")
print()
print("  注意 <|im_start|> / <|im_end|> 这些是特殊 token，不是普通文字。")
print("  它们告诉模型'这段是用户说的''该你回答了'。")

# tokenizer 把文本切成 token 并查表得到 id。
# return_tensors="pt" 让它直接返回 PyTorch tensor 而不是 Python list。
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.device)

print()
print(f"input_ids.shape  = {tuple(input_ids.shape)}   # [batch=1, seq_len={input_ids.shape[1]}]")
print(f"input_ids.dtype  = {input_ids.dtype}          # id 是整数，不是浮点")
print(f"input_ids.device = {input_ids.device}")

print()
print("逐个看看这句话被切成了什么（前 25 个）:")
print(f"  {'序号':>4}  {'token id':>9}  文本")
for i, tid in enumerate(input_ids[0].tolist()[:25]):
    # 单个 id 反解成文本。repr 是为了让换行符 \n 显示出来而不是真的换行。
    piece = tokenizer.decode([tid])
    print(f"  {i:>4}  {tid:>9}  {piece!r}")

print()
print("回答第一个问题：文本在 tokenizer(...) 这一步变成 token id，")
print("在这之后，模型内部再也见不到'文字'，只有整数和浮点数。")


# ---------------------------------------------------------------------------
banner("2. 一次 forward")
# ---------------------------------------------------------------------------

# inference_mode：不记录反向传播信息（00a 第 7 节）
with torch.inference_mode():
    # 先空跑一次做预热。第一次调用要初始化 CUDA 上下文、编译/挑选 kernel、
    # 分配显存池，耗时能到几百毫秒，完全不能代表真实速度。
    with Timer() as t_cold:
        model(input_ids=input_ids)

    with Timer() as t:
        outputs = model(input_ids=input_ids)

print(f"第一次 forward（含预热）: {t_cold.elapsed*1000:7.2f} ms")
print(f"第二次 forward（真实）  : {t.elapsed*1000:7.2f} ms")
print(f"预热开销占了 {(1 - t.elapsed/t_cold.elapsed)*100:.0f}%，"
      "所以任何性能测量都要先预热再计时。")
print()
print(f"返回对象类型: {type(outputs).__name__}")
print(f"里面有哪些字段: {[k for k, v in outputs.items()]}")

logits = outputs.logits
print()
print(f"logits.shape  = {tuple(logits.shape)}")
print(f"              = [batch=1, seq_len={logits.shape[1]}, vocab_size={logits.shape[2]}]")
print(f"logits.dtype  = {logits.dtype}")
print(f"logits.device = {logits.device}")

print()
print("回答第二个问题：")
print("  输入 = 一串 token id，形状 [batch, seq_len]")
print("  输出 = 对词表里每个词的打分，形状 [batch, seq_len, vocab_size]")
print()
print(f"  这次一共产生了 {logits.numel():,} 个浮点数，"
      f"占 {logits.numel() * logits.element_size() / 1024**2:.1f} MB。")
print("  记住这个数，第 3 节会说明其中绝大部分是白算的。")


# ---------------------------------------------------------------------------
banner("3. 为什么只看最后一个位置")
# ---------------------------------------------------------------------------

print("logits 有 seq_len 行，每一行是'看完这个位置为止的内容后，下一个词该是什么'。")
print()
print("挑几个位置看看模型分别在预测什么:")
print(f"  {'位置':>4}  {'该位置输入的词':<14}  {'模型预测下一个词'}")

for pos in [0, 1, len(input_ids[0]) // 2, -2, -1]:
    cur_id = input_ids[0, pos].item()
    pred_id = logits[0, pos].argmax(dim=-1).item()
    print(f"  {pos:>4}  {tokenizer.decode([cur_id])!r:<14}  "
          f"{tokenizer.decode([pred_id])!r}")

print()
print("前面那些位置的预测，其实是在'预测我们已经知道的内容'——")
print("位置 0 预测位置 1，可位置 1 的词我们输入里本来就有。")
print("只有最后一个位置的预测，指向的是我们还不知道的、真正要生成的下一个词。")
print()
print("回答第三个问题：前面 seq_len-1 行在生成时用不上，所以只取 [:, -1, :]。")

# 真实推理框架会把这件事做成一个开关，避免白算
print()
print("transformers v5 把这个优化做成了 logits_to_keep 参数：")

with torch.inference_mode():
    out_keep1 = model(input_ids=input_ids, logits_to_keep=1)

print(f"  默认             logits.shape = {tuple(logits.shape)}")
print(f"  logits_to_keep=1 logits.shape = {tuple(out_keep1.logits.shape)}")
saved = 1 - out_keep1.logits.numel() / logits.numel()
print(f"  省掉了 {saved*100:.1f}% 的 lm_head 计算量和输出显存")
print()
print("  `generate()` 内部就是这么做的。阶段 0 为了看得清楚，我们仍用默认的全量输出。")

# 顺带演示 bf16 的数值不确定性，这个坑 00d 会再遇到
same = torch.equal(logits[:, -1, :], out_keep1.logits[:, -1, :])
maxdiff = (logits[:, -1, :].float() - out_keep1.logits[:, -1, :].float()).abs().max()
print()
print(f"  顺带一提：两种算法最后一行完全相等吗 -> {same}")
print(f"  最大差异 {maxdiff.item():.4f}，argmax 仍然一致。")
print("  bf16 下改变矩阵形状会走不同的 kernel，结果有微小浮动，这是正常的。")
print("  所以 00d 对比时要比 token id，不要比 logits 数值。")


# ---------------------------------------------------------------------------
banner("4. 从 logits 到下一个 token")
# ---------------------------------------------------------------------------

# 取最后一个位置。[0] 取 batch 里第一条，[-1] 取最后一个位置
last_logits = logits[0, -1, :]
print(f"last_logits.shape = {tuple(last_logits.shape)}   # 词表里每个词一个分数")
print(f"分数范围: {last_logits.min().item():.2f} ~ {last_logits.max().item():.2f}")

# greedy：直接选分数最高的
next_id = last_logits.argmax(dim=-1).item()
print()
print(f"argmax 选出的 token id = {next_id}")
print(f"对应文本 = {tokenizer.decode([next_id])!r}")

# 看看排名靠前的几个候选，理解"为什么还有采样这回事"
probs = torch.softmax(last_logits.float(), dim=-1)
top = torch.topk(probs, 5)
print()
print("排名前 5 的候选:")
print(f"  {'排名':>4}  {'token id':>9}  {'概率':>8}  文本")
for rank, (p, tid) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
    print(f"  {rank:>4}  {tid:>9}  {p*100:>7.2f}%  {tokenizer.decode([tid])!r}")

print()
print("回答第四个问题：logits 经过 argmax（greedy）或 softmax + 抽样，变成一个 token id。")
print("阶段 0 只用 argmax，因为它没有随机性，方便和 generate() 逐字对比。")


# ---------------------------------------------------------------------------
banner("小结：我们现在只前进了一个 token")
# ---------------------------------------------------------------------------

print(f"""
输入 {input_ids.shape[1]} 个 token，跑一次 forward，得到了第 1 个新 token: \
{tokenizer.decode([next_id])!r}

想要第 2 个 token，就得把它拼回输入，凑成 {input_ids.shape[1] + 1} 个 token 再跑一次。
想要第 20 个，就得这样重复 20 次。

「一次 forward 只产出一个 token」——这句话是整个推理优化领域的起点。
后面所有的 KV Cache、continuous batching、投机解码，
都是在想办法让这个重复过程变便宜。

下一步: 00c_manual_decode.py 把这个循环真正写出来。
""")


banner("动手改改看")
print("""
1. 把 common.py 里 build_prompt 的 enable_thinking 改成 True，
   重新跑这个脚本，看看 prompt 文本多了什么，input_ids 变长了多少。

2. 不套对话模板，直接把 DEFAULT_QUESTION 原文喂进去（跳过 build_prompt），
   看看 argmax 选出来的第一个词变成了什么。能解释为什么吗？

3. 把第 4 节的 argmax 换成按概率抽样:
       next_id = torch.multinomial(probs, num_samples=1).item()
   多跑几次，结果还一样吗？
""")
