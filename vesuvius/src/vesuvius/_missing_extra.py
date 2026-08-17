"""Placeholder for a public name whose optional extra is not installed.

Guarded imports used to fall back to ``None``, which meant the first thing a caller
saw was ``TypeError: 'NoneType' object is not callable`` raised at their own call
site, with the ImportError that actually explained it already discarded two frames
deeper. ``MissingExtra`` keeps the name present and falsy, so ``hasattr`` and
``if not vesuvius.models`` behave exactly as before, and reports the missing module
plus the extra that supplies it as soon as the name is used.
"""

from __future__ import annotations


class MissingExtra:
    """Falsy stand-in that raises an explanatory ImportError when used."""

    __slots__ = ("_name", "_extra", "_cause")

    def __init__(self, name: str, extra: str, cause: BaseException) -> None:
        self._name = name
        self._extra = extra
        self._cause = cause

    def _fail(self) -> None:
        raise ImportError(
            f"vesuvius.{self._name} requires the '{self._extra}' extra, which is not "
            f"installed: {self._cause}. Install it with "
            f'`pip install "vesuvius[{self._extra}]"`.'
        ) from self._cause

    def __call__(self, *_args, **_kwargs):
        self._fail()

    def __getattr__(self, _attr: str):
        self._fail()

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return (
            f"<vesuvius.{self._name} unavailable: '{self._extra}' extra not installed>"
        )


__all__ = ["MissingExtra"]
