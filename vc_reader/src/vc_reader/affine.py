"""3x4 affine transforms in the metadata's ``[x, y, z]`` convention.

Arithmetic uses the native catalog implementation shared with C++ clients.
Matrices retain float64 coefficients from the published registrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from . import _catalog

Row = tuple[float, float, float, float]
Point = tuple[float, float, float]


@dataclass(frozen=True)
class Affine:
    """``p' = R @ p + t`` on column vectors ``[x, y, z, 1]``.

    ``matrix`` holds the three rows of ``[R | t]``. Compose with ``@``:
    ``(a @ b).apply(p) == a.apply(b.apply(p))``.
    """

    matrix: tuple[Row, Row, Row]
    derivation_path: str | None = None

    @classmethod
    def from_rows(cls, rows: Sequence[Sequence[float]], derivation_path: str | None = None) -> Affine:
        if len(rows) != 3 or any(len(r) != 4 for r in rows):
            raise ValueError(f"expected a 3x4 matrix, got {rows!r}")
        return cls(tuple(tuple(float(v) for v in r) for r in rows), derivation_path)  # type: ignore[arg-type]

    @classmethod
    def identity(cls) -> Affine:
        return cls.scale(1.0)

    @classmethod
    def scale(cls, s: float) -> Affine:
        """Uniform scale, e.g. ``Affine.scale(2**level)`` maps level ``level`` voxels to level 0."""
        s = float(s)
        return cls(((s, 0.0, 0.0, 0.0), (0.0, s, 0.0, 0.0), (0.0, 0.0, s, 0.0)))

    @property
    def linear(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(r[:3] for r in self.matrix)

    @property
    def translation(self) -> Point:
        return (self.matrix[0][3], self.matrix[1][3], self.matrix[2][3])

    def apply(self, point: Sequence[float]) -> Point:
        return tuple(_catalog.affine_apply(self.matrix, tuple(point)))

    def __matmul__(self, other: Affine) -> Affine:
        if not isinstance(other, Affine):
            return NotImplemented
        return Affine.from_rows(_catalog.affine_compose(self.matrix, other.matrix))

    def inverse(self) -> Affine:
        return Affine.from_rows(_catalog.affine_inverse(self.matrix))

