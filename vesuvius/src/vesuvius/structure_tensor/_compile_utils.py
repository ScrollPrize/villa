import os
import sys
import warnings
from typing import Callable, Optional

import torch

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _maybe_compile_function(
    fn: Callable,
    *,
    compile_kwargs: Optional[dict] = None,
) -> Callable:
    """
    Attempt to wrap ``fn`` with ``torch.compile`` while guarding for environments
    that cannot support the compilation toolchain (e.g., Windows without MSVC).
    Returns the original function if compilation is unavailable or disabled.
    """
    torch_compile = getattr(torch, "compile", None)
    if torch_compile is None:
        return fn

    disable_env = os.environ.get("VESUVIUS_DISABLE_TORCH_COMPILE", "")
    if disable_env and disable_env.lower() in _TRUTHY:
        return fn

    legacy_env = os.environ.get("VESUVIUS_TORCH_COMPILE")
    if legacy_env and legacy_env.lower() in _FALSY:
        return fn

    if sys.platform.startswith("win"):
        return fn

    kwargs = {"mode": "reduce-overhead", "fullgraph": True}
    if compile_kwargs:
        kwargs.update(compile_kwargs)

    try:
        return torch_compile(fn, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive fallback
        warnings.warn(
            f"torch.compile failed; falling back to eager execution: {exc}",
            RuntimeWarning,
        )
        return fn
