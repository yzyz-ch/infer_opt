"""05 这一版浪费在哪。"""


def run(records):
    print("##### 5. 这一版浪费在哪 ###")
    n = len(records)
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

下一步: 00d_kv_cache/ 把它实现出来，并回答"到底什么时候才划算"。
""")

    print("##### 动手改改看 ###")
    print("""
1. 把 greedy_loop.py 里"拼回输入"那两行注释掉，看看会输出什么。
   （提示：模型每轮看到的上下文都一样）

2. 把 argmax 换成随机抽样:
       probs = torch.softmax(next_token_logits.float(), dim=-1)
       next_id = torch.multinomial(probs, num_samples=1).item()
   跑三次，结果还一样吗？这就是 do_sample=True 在做的事。

3. 把 MAX_NEW_TOKENS 改成 5，观察最后打印的是"正常收尾"还是"强制截断"。

4. 把 logits[0, -1, :] 改成 logits[0, 0, :]（取第一个位置），
   生成的文字会变成什么样？为什么？

5. 04 的扫描里把 logits_to_keep=1 去掉，重新跑。
   4096 那一行的耗时变化大吗？（提示：lm_head 要输出 4096x151936 个数）
""")

if __name__ == "__main__":
    from importlib import import_module

    _tok, _model, _ids, records = import_module("01_run_loop").run()
    run(records)
