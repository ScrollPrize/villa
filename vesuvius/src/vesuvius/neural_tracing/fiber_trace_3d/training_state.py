"""Shared durable snapshot and in-RAM cohort handoff state.

Capturing a handoff is not a public training-step snapshot: the caller owns this
CPU payload until replacement workers have restored it. Dataset frontiers and
test/logging progress belong in ``progress``, never reconstructed from a rank's
physical microbatch count.
"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class TrainingBoundary:
    """A completed logical update/test boundary, not a physical microbatch."""

    step: int
    done: bool
    capture: Any
    phase: str = "update"


def config_fingerprint(config):
    """Identify the numerical/data config, excluding manager control settings."""
    config = deepcopy(config)
    config.setdefault("training", {}).pop("cooperative", None)
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def model_optimizer_payload(model, optimizer, grad_scaler=None, ema=None):
    """Common payload; callers choose disk serialization or CPU RAM ownership."""
    result = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    if grad_scaler is not None and grad_scaler.is_enabled():
        result["grad_scaler"] = grad_scaler.state_dict()
    if ema is not None and ema.tracks:
        result["ema"] = ema.state_dict()
    return result


def restore_model_optimizer(payload, *, model, optimizer=None, grad_scaler=None):
    if optimizer is not None and payload.get("snapshot_kind") == "ema":
        raise ValueError("EMA snapshots are inference-only; resume from the raw current.pt snapshot")
    model.load_state_dict(payload.get("model", payload))
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if grad_scaler is not None and grad_scaler.is_enabled() and "grad_scaler" in payload:
        grad_scaler.load_state_dict(payload["grad_scaler"])


def cpu_clone(value):
    """Detach all tensors and copy storage, including already-CPU tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        result = type(value)((key, cpu_clone(item)) for key, item in value.items())
        if isinstance(value, OrderedDict) and hasattr(value, "_metadata"):
            result._metadata = deepcopy(value._metadata)
        return result
    if isinstance(value, tuple):
        return tuple(cpu_clone(item) for item in value)
    if isinstance(value, list):
        return [cpu_clone(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.copy()
    if value is None or isinstance(value, (str, bytes, bool, int, float, np.generic)):
        return deepcopy(value)
    raise TypeError(f"unsupported handoff state value {type(value).__name__}")


def capture_handoff(*, model, optimizer, progress, grad_scaler=None, cuda_device=None, ema=None):
    """Called at a completed unit boundary after stopping asynchronous prep.

    CUDA RNG is accessed only when the GPU worker explicitly supplies its own
    device. A CPU-only caller never probes CUDA availability or initializes it.
    """
    payload = cpu_clone(model_optimizer_payload(model, optimizer, grad_scaler, ema))
    payload["progress"] = cpu_clone(progress)
    payload["rng"] = {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }
    if cuda_device is not None:
        payload["rng"]["cuda"] = torch.cuda.get_rng_state(cuda_device).cpu().clone()
    return payload


def restore_handoff(payload, *, model, optimizer, grad_scaler=None, cuda_device=None):
    if "progress" not in payload or "rng" not in payload:
        raise ValueError("incomplete in-RAM training handoff")
    # Optimizer.load_state_dict may reuse CPU tensor storage. Keep the parent's
    # handoff immutable even in CPU equivalence tests / CPU restore paths.
    restore_model_optimizer(cpu_clone(payload), model=model, optimizer=optimizer, grad_scaler=grad_scaler)
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if cuda_device is not None:
        if "cuda" not in rng:
            raise ValueError("GPU training handoff has no CUDA RNG state")
        torch.cuda.set_rng_state(rng["cuda"], cuda_device)
    elif "cuda" in rng:
        raise ValueError("restoring GPU handoff requires an explicit worker device")
    return cpu_clone(payload["progress"])


def replica_signature(payload):
    """Check model, optimizer, scaler and logical progress, excluding rank RNG.

    Hashing happens only at handoff, not per training update. CPU tensors are
    required so this cannot silently create transfers in a supervisor.
    """
    digest = hashlib.sha256()

    def visit(value: Any):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or value.layout != torch.strided:
                raise ValueError("replica signature requires dense CPU tensors")
            digest.update(f"tensor:{value.dtype}:{tuple(value.shape)}:".encode())
            digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"{")
            for key in sorted(value, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(value[key])
            digest.update(b"}")
        elif isinstance(value, (tuple, list)):
            digest.update(type(value).__name__.encode() + b"[")
            for item in value:
                visit(item)
            digest.update(b"]")
        elif value is None or isinstance(value, (str, bytes, bool, int, float)):
            data = repr(value).encode()
            digest.update(f"{type(value).__name__}:{len(data)}:".encode() + data)
        else:
            raise TypeError(f"unsupported replica signature value {type(value).__name__}")

    visit({key: value for key, value in payload.items() if key != "rng"})
    return digest.hexdigest()
