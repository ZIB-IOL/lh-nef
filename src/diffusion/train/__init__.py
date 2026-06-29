"""Trainer registry hooks for HiP-token diffusion."""

from __future__ import annotations

from . import extract_trainer  # noqa: F401
from . import dm_trainer  # noqa: F401
from . import sample_trainer  # noqa: F401

__all__ = ["extract_trainer", "dm_trainer", "sample_trainer"]

