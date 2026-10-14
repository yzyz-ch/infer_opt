"""04 停止条件写错会怎样。"""

import torch


def run(tokenizer, model, prompt_ids, eos_ids):
    print("##### 4. 停止条件写错会怎样 ###")
    print("先看看这个模型的结束符到底是怎么配置的：")
    print(f"  config.json          里 eos_token_id = {model.config.eos_token_id}")
    print(
        f"  generation_config    里 eos_token_id = "
        f"{model.generation_config.eos_token_id}"
    )
    print(f"  tokenizer            里 eos_token_id = {tokenizer.eos_token_id}")
    print()
    print("  三处不一样！generation_config 里是个【列表】，另外两处是单个整数。")
    print("  真正生效的是 generation_config，因为 generate() 用的就是它。")
    print()
    print(
        f"  列表里两个 id 分别是: "
        f"{[(i, tokenizer.decode([i])) for i in model.generation_config.eos_token_id]}"
    )

    print()
    print("现在故意写错停止条件，看看后果：")
    print("  错误写法: if next_id == model.generation_config.eos_token_id")
    print("  这是拿一个 int 去和一个 list 比，永远返回 False。")
    print()

    wrong_stop = model.generation_config.eos_token_id
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
            if nid == wrong_stop:
                break

    print(
        f"  本该在第 {stop_should_have_been} 步停下，实际跑满了 {len(buggy_out)} 步。"
    )
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

if __name__ == "__main__":
    from importlib import import_module

    ctx = import_module("01_run_three").run()
    run(ctx["tokenizer"], ctx["model"], ctx["prompt_ids"], ctx["eos_ids"])
