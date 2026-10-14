"""05 generate() 替你做了什么，以及自查清单。"""


def run():
    print("##### 5. generate() 到底替你做了什么 ###")
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

    print("##### 阶段 0 完成标准自查 ###")
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
    run()
