"""Tier 4 -- frozen-checkpoint inference regression.

For each entry in `tests/regression_baselines.yaml`, runs `src/eval_ckpt.py`
on the recorded checkpoint and asserts that the reported metric matches the
recorded `expected` value within `tol`.

Entries with empty `ckpt` or non-existing files are skipped so this test file
is safe to commit before any baselines are recorded.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES = REPO_ROOT / "tests" / "regression_baselines.yaml"
EVAL_SCRIPT = REPO_ROOT / "src" / "eval_ckpt.py"


def _load_baselines() -> list[tuple[str, dict]]:
    if not BASELINES.is_file():
        return []
    with BASELINES.open("r") as f:
        data = yaml.safe_load(f) or {}
    return list(data.items())


def _parse_metric(stdout: str, metric: str) -> float | None:
    # eval_ckpt.py emits e.g. "[eval] split=test ..., psnr=27.6543, mse=..."
    m = re.search(rf"\b{re.escape(metric)}=([\-+]?\d+(?:\.\d+)?(?:[eE][\-+]?\d+)?)", stdout)
    return float(m.group(1)) if m else None


@pytest.mark.gpu
@pytest.mark.regression
@pytest.mark.parametrize("name,entry", _load_baselines(),
                        ids=[n for n, _ in _load_baselines()] or ["__nobaselines__"])
def test_frozen_checkpoint_metric(name: str, entry: dict):
    if not entry or not entry.get("ckpt"):
        pytest.skip(f"baseline {name!r}: ckpt not configured")

    ckpt = Path(str(entry["ckpt"])).expanduser()
    if not ckpt.exists():
        pytest.skip(f"baseline {name!r}: ckpt path does not exist ({ckpt})")

    split = str(entry.get("split", "test"))
    metric = str(entry.get("metric", "psnr"))
    expected = float(entry["expected"])
    tol = float(entry.get("tol", 0.1))

    # eval_ckpt.py accepts a RUN_DIR (containing cfg.yaml + best-model.pth) or a .pth via --ckpt.
    if ckpt.is_dir():
        cmd = [sys.executable, str(EVAL_SCRIPT), "--run_dir", str(ckpt), "--split", split, "--no_vis"]
    else:
        cmd = [sys.executable, str(EVAL_SCRIPT), "--ckpt", str(ckpt), "--split", split, "--no_vis"]

    env = os.environ.copy()
    env.setdefault("LHNEF_SAVE_ROOT", "/tmp/lhnef_test_save")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert proc.returncode == 0, (
        f"eval_ckpt.py exited {proc.returncode} for {name!r}\n"
        f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    )

    # eval_ckpt.py logs via Python's `logging` module (writes to stderr),
    # so parse both streams to be robust to where the metric line lands.
    got = _parse_metric(proc.stdout + "\n" + proc.stderr, metric)
    assert got is not None, (
        f"could not parse {metric!r} from eval_ckpt.py output for {name!r}\n"
        f"stdout tail:\n{proc.stdout[-2000:]}\n"
        f"stderr tail:\n{proc.stderr[-2000:]}"
    )

    diff = abs(got - expected)
    assert diff <= tol, (
        f"{name!r}: {metric}={got:.4f} drifted from expected {expected:.4f} "
        f"by {diff:.4f} (tol={tol:.4f})"
    )
