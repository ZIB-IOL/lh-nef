# Preload cv2 *before* any module that imports PIL/torch, so cv2 wins the race
# for the conda-forge libjpeg/libtiff load. Without this, PIL grabs a libjpeg
# without jpeg12 symbols and cv2's libtiff fails on later import.
# In envs without cv2/ffcv installed, this is a soft no-op.
try:
    import cv2  # noqa: F401
except Exception:
    pass

from .datasets import register, make
from . import wrappers
from . import cifar10
from . import imagenet_labeled

# FFCV-backed loader for ImageNet 256x256 (optional; requires ffcv installed).
# Wrapped in try/except so the codebase still imports cleanly in environments
# without the ffcv stack.
try:
    from . import ffcv_imagenet256  # noqa: F401
except Exception as _e:  # pragma: no cover
    pass
from . import celebahq
from . import shapenet16_vox_occ
from . import era5_temperature

try:
    import diffusion.data  # noqa: F401
except Exception:
    pass

try:
    import classification.data  # noqa: F401
except Exception:
    pass

try:
    import era5_forecasting.data  # noqa: F401
except Exception:
    pass
