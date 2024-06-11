import importlib.metadata
__version__ = importlib.metadata.version("dasheng")

from .pretrained.pretrained import dasheng_base, dasheng_06B, dasheng_12B
