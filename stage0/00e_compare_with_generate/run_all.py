"""按顺序跑完 00e。在 stage0 目录下: python 00e_compare_with_generate/run_all.py"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _path  # noqa: F401

from importlib import import_module

ctx = import_module("01_run_three").run()
d_nc, _d_c = import_module("02_compare").run(
    ctx["tokenizer"],
    ctx["ids_nocache"],
    ctx["ids_cache"],
    ctx["ids_generate"],
)
import_module("03_inspect_fork").run(
    ctx["tokenizer"],
    ctx["model"],
    ctx["prompt_ids"],
    ctx["ids_nocache"],
    d_nc,
)
import_module("04_wrong_eos").run(
    ctx["tokenizer"],
    ctx["model"],
    ctx["prompt_ids"],
    ctx["eos_ids"],
)
import_module("05_generate_and_checklist").run()
