"""Helpers for the lab tests.

Each lab folder has ``lab.py`` (yours, with TODOs), ``solution.py`` (the reference) and
``test_lab.py``. The tests check ``lab.py`` by default; set ``LAB_TARGET=solution`` to check the
reference instead:

    pytest labs/module-01/lesson-01                    # checks your lab.py
    LAB_TARGET=solution pytest labs/module-01/lesson-01
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def load_target(test_file: str):
    """Import ``lab.py`` (or ``$LAB_TARGET.py``) from the folder that contains ``test_file``."""
    folder = Path(test_file).resolve().parent
    name = os.environ.get("LAB_TARGET", "lab")
    path = folder / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{folder.parent.name}_{folder.name}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_path(path: str):
    """Import a Python file by path (used by worker processes that must load the learner's lab.py)."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"lab_{abs(hash(str(path)))}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def target_path(test_file: str) -> str:
    """Path of lab.py (or $LAB_TARGET.py) next to ``test_file``, for passing to worker processes."""
    folder = Path(test_file).resolve().parent
    return str(folder / f"{os.environ.get('LAB_TARGET', 'lab')}.py")
