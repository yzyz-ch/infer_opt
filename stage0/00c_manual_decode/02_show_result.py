"""02 看生成结果。"""


def run(tokenizer, generated_ids):
    print("##### 2. 生成结果 ###")
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(text)
    print()
    print("带特殊标记的原始版本（能看到结尾的 <|im_end|>）:")
    print(f"  {tokenizer.decode(generated_ids, skip_special_tokens=False)!r}")

if __name__ == "__main__":
    from importlib import import_module

    tok, _model, ids, _records = import_module("01_run_loop").run()
    run(tok, ids)
