import importlib

from . import case, curved_divider_wall, mesh

__all__ = ["case", "mesh", "curved_divider_wall"]


def __getattr__(name):
    if name in __all__:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
