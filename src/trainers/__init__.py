from .trainers import register, trainers_dict
from . import base_trainer
from . import lhnef_trainer

# Back-compat alias for legacy cfg.yaml files using trainer='infd_trainer'.
trainers_dict['infd_trainer'] = trainers_dict['lhnef_trainer']

try:
    import diffusion.train  # noqa: F401
except Exception:
    pass

try:
    import classification.train  # noqa: F401
except Exception:
    pass

try:
    import era5_forecasting.train  # noqa: F401
except Exception:
    pass
