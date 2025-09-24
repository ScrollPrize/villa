"""Training utilities exposed via lazy imports to avoid circular dependencies."""

from typing import Any, Callable, Dict

__all__ = ["BaseTrainer", "SurfaceFrameTrainer"]

def _load_base_trainer() -> Any:
    from .train import BaseTrainer

    return BaseTrainer


def _load_surface_frame_trainer() -> Any:
    from .surface_frame_trainer import SurfaceFrameTrainer

    return SurfaceFrameTrainer


_LOADERS: Dict[str, Callable[[], Any]] = {
    "BaseTrainer": _load_base_trainer,
    "SurfaceFrameTrainer": _load_surface_frame_trainer,
}


def __getattr__(name: str) -> Any:
    try:
        loader = _LOADERS[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise AttributeError(name) from exc
    return loader()


def __dir__() -> list[str]:  # pragma: no cover - convenience
    return sorted({*globals().keys(), *_LOADERS})
