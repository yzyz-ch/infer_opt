"""06 小结与练习。"""


def run(tokenizer, input_ids, next_id):
    print("##### 小结：我们现在只前进了一个 token ###")
    print(f"""
输入 {input_ids.shape[1]} 个 token，跑一次 forward，得到了第 1 个新 token: \
{tokenizer.decode([next_id])!r}

想要第 2 个 token，就得把它拼回输入，凑成 {input_ids.shape[1] + 1} 个 token 再跑一次。
想要第 20 个，就得这样重复 20 次。

「一次 forward 只产出一个 token」——这句话是整个推理优化领域的起点。
后面所有的 KV Cache、continuous batching、投机解码，
都是在想办法让这个重复过程变便宜。

下一步: 00c_manual_decode/ 把这个循环真正写出来。
""")

    print("##### 动手改改看 ###")
    print("""
1. 把 common.py 里 build_prompt 的 enable_thinking 改成 True，
   重新跑 02_tokenize.py，看看 prompt 文本多了什么，input_ids 变长了多少。

2. 不套对话模板，直接把 DEFAULT_QUESTION 原文喂进去（跳过 build_prompt），
   看看 argmax 选出来的第一个词变成了什么。能解释为什么吗？

3. 把 05 里的 argmax 换成按概率抽样:
       next_id = torch.multinomial(probs, num_samples=1).item()
   多跑几次，结果还一样吗？
""")

if __name__ == "__main__":
    from importlib import import_module

    tok, model = import_module("01_load").run()
    input_ids = import_module("02_tokenize").run(tok, model)
    logits = import_module("03_one_forward").run(model, input_ids)
    next_id = import_module("05_next_token").run(tok, logits)
    run(tok, input_ids, next_id)
