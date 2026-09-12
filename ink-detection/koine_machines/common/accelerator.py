"""Accelerator selection for ink-detection inference.

The main ``vesuvius`` package has resolved cuda -> mps -> cpu for a while
(``vesuvius/src/vesuvius/models/utilities/get_accelerator.py``), and nnU-Net
under ``segmentation/models/arch/nnunet`` handles MPS too. The two
``koine_machines.inference`` entrypoints predated that and only ever looked
for CUDA, so on Apple Silicon they fell through to the CPU branch and ran the
published ``ink_9um`` checkpoints on the CPU with the GPU idle -- no error, no
warning, just far slower than the hardware allows.

Everything here is deliberately narrow: pick a device, and say whether that
device has an autocast path worth taking.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

LOGGER = logging.getLogger(__name__)

DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")

# Backends whose autocast we have actually exercised for inference.
#
# CPU is excluded on purpose. torch CPU autocast is bfloat16-only, and on the
# machines that reach the CPU branch at all it is not reliably a win over plain
# float32, so enabling it there would trade correctness margin for nothing.
#
# MPS is included: measured on an M5 Pro / macOS 26.5.1 / torch 2.14.0,
# ``torch.autocast(device_type="mps")`` runs for dtype float16, bfloat16 and
# the backend default (which resolves to float16).
AUTOCAST_DEVICE_TYPES = frozenset({"cuda", "mps"})


def mps_available() -> bool:
    """True when this torch build can actually reach Metal."""
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_available())
    except Exception:  # pragma: no cover - defensive, older/odd torch builds
        return False


def available_device_types() -> tuple[str, ...]:
    types: list[str] = []
    if torch.cuda.is_available():
        types.append("cuda")
    if mps_available():
        types.append("mps")
    types.append("cpu")
    return tuple(types)


def select_device(
    preference: str = "auto",
    *,
    gpu_ids: Sequence[int] = (),
) -> torch.device:
    """Resolve the inference device.

    ``gpu_ids`` comes from ``--gpus`` and is CUDA-only by construction (it
    indexes CUDA ordinals and feeds ``nn.DataParallel``), so passing it keeps
    the original hard error when CUDA is absent rather than quietly running
    somewhere the user did not ask for.
    """
    gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids)
    preference = str(preference or "auto").strip().lower()
    if preference not in DEVICE_CHOICES:
        raise ValueError(
            f"Unsupported device {preference!r}; choose one of {', '.join(DEVICE_CHOICES)}."
        )

    if gpu_ids:
        if preference not in {"auto", "cuda"}:
            raise ValueError(
                f"--gpus selects CUDA ordinals but --device={preference} was requested."
            )
        if not torch.cuda.is_available():
            raise ValueError("--gpus was provided, but CUDA is not available in this environment.")
        available_gpu_count = int(torch.cuda.device_count())
        invalid_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id >= available_gpu_count]
        if invalid_gpu_ids:
            raise ValueError(
                f"Requested CUDA device ids {invalid_gpu_ids!r} are unavailable; "
                f"visible device count is {available_gpu_count}."
            )
        return torch.device(f"cuda:{gpu_ids[0]}")

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device=cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    if preference == "mps":
        if not mps_available():
            raise ValueError("--device=mps was requested, but MPS is not available.")
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    LOGGER.warning(
        "No GPU backend found (CUDA and MPS both unavailable); running inference on the CPU. "
        "This is correct but slow -- pass --device explicitly to silence this."
    )
    return torch.device("cpu")


def autocast_supported(device: torch.device) -> bool:
    return device.type in AUTOCAST_DEVICE_TYPES


def log_device(device: torch.device) -> None:
    if device.type == "cuda":
        LOGGER.info("Inference device: %s (%s)", device, torch.cuda.get_device_name(device))
    elif device.type == "mps":
        try:
            budget = torch.mps.recommended_max_memory() / 2**30
        except Exception:  # pragma: no cover - not on every torch build
            LOGGER.info("Inference device: mps (Apple Silicon)")
        else:
            LOGGER.info(
                "Inference device: mps (Apple Silicon, recommended max %.1f GiB)", budget
            )
    else:
        LOGGER.info("Inference device: %s", device)
