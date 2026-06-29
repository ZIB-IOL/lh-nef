from .models import register, make, models as _model_registry
from . import lhnef
from . import renderers
from . import hip_coord_value_encoder

_model_registry['infd'] = _model_registry['lhnef']

try:
    import diffusion.models  # noqa: F401
except Exception:
    pass

try:
    import classification.models  # noqa: F401
except Exception:
    pass

try:
    import era5_forecasting.models  # noqa: F401
except Exception:
    pass
