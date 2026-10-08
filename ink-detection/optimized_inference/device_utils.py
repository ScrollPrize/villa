"""Device selection and accelerator helpers.

Kept in one place so a new backend is added once rather than at every call
site. Before this existed, the device was chosen as
``cuda if torch.cuda.is_available() else cpu``, which silently sent every
Apple Silicon machine down the CPU path, and the autocast device type was
chosen as ``"cuda" if device.type == "cuda" else "cpu"``, which makes
autocast a no-op when the tensors are on MPS.
"""
import os
import torch

__all__ = ["select_device", "amp_device_type", "sync", "empty_cache"]


def select_device(prefer: str | None = None) -> torch.device:
    """cuda, then mps, then cpu. ``INK_DEVICE`` or *prefer* overrides."""
    want = (prefer or os.getenv("INK_DEVICE") or "").strip().lower()
    if want:
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_device_type(device: torch.device) -> str:
    """The device_type torch.autocast should be given for *device*.

    Must follow the device the tensors actually live on. Passing "cpu" while
    the tensors are on MPS does not error, it silently disables autocast.
    """
    t = device.type if isinstance(device, torch.device) else str(device)
    return t if t in ("cuda", "mps", "cpu", "xpu") else "cpu"


def sync(device: torch.device) -> None:
    """Block until queued work on *device* has finished, for honest timing."""
    t = device.type if isinstance(device, torch.device) else str(device)
    if t == "cuda":
        torch.cuda.synchronize()
    elif t == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device | None = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()
