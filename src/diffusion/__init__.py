"""
LH-NeF downstream diffusion: HiP-token diffusion (Stage 2A).

Integrates with LH-NeF's registry-based configuration system:
- Models register via the global `models.register` decorator.
- Datasets register via the global `datasets.register` decorator.
- Trainers register via the global `trainers.register` decorator.

Importing this package is side-effect free; importing its submodules
(`diffusion.models`, `diffusion.data`, `diffusion.train`) populates registries.
"""

from __future__ import annotations

__all__ = []
