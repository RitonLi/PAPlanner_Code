import os

from qwen3_vl.utils import io

__all__ = ["EVAL_ROOT", "TASKS"]


EVAL_ROOT = "scripts/v3/eval"
TASKS = io.load(os.path.join(os.path.dirname(__file__), "registry.yaml"))
