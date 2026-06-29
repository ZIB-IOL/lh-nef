"""Shared fixtures and path setup for LH-NeF tests.

`src/` is added to `sys.path` (mirroring `run.py`) so tests can `import models`,
`import datasets`, etc. just like the entry points do.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for p in (SRC_ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("LHNEF_DATA_ROOT", str(REPO_ROOT / "load"))
os.environ.setdefault("LHNEF_SAVE_ROOT", "/tmp/lhnef_test_save")

import pytest  # noqa: E402
import torch  # noqa: E402


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


def pytest_collection_modifyitems(config, items):
    """Auto-skip @pytest.mark.gpu tests when CUDA is unavailable."""
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
