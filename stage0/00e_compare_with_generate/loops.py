"""从 00c / 00d 目录加载解码函数，避免和同名 .py 入口冲突。"""

import importlib.util
from pathlib import Path

STAGE0 = Path(__file__).resolve().parent.parent


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_loops():
    c = _load(STAGE0 / "00c_manual_decode" / "greedy_loop.py")
    d = _load(STAGE0 / "00d_kv_cache" / "cached_loop.py")
    return c, d
