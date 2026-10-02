"""Placeholders for names whose optional dependencies are not installed.

Binding such a name to None makes using it fail with
"'NoneType' object is not callable", which names neither the missing
dependency nor the extra that provides it. These placeholders raise
ImportError saying what to install, and keep the original failure as
``__cause__``.
"""

from typing import Any, Callable


def requires_extra(name: str, extra: str, cause: BaseException) -> Callable[..., Any]:
    """Return a stand-in for `name` that explains which extra provides it."""

    def _missing(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            f"vesuvius.{name} needs optional dependencies that are not installed. "
            f"Install them with: pip install 'vesuvius[{extra}]'"
        ) from cause

    _missing.__name__ = name
    _missing.__qualname__ = name
    _missing.__doc__ = f"Unavailable: install vesuvius[{extra}] to use {name}."
    return _missing
