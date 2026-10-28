"""3x4 affine transforms in the metadata's ``[x, y, z]`` convention.

Pure Python so that the parsing layer has no dependencies. Matrices are kept
exactly as they appear in ``metadata.json``; the only arithmetic done on them
by vc_reader itself is composition with power-of-two pyramid-level scales,
which is exact in floating point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

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
        x, y, z = point
        m = self.matrix
        return (
            m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
            m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
            m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3],
        )

    def apply_many(self, points: Iterable[Sequence[float]]) -> list[Point]:
        return [self.apply(p) for p in points]

    def __matmul__(self, other: Affine) -> Affine:
        if not isinstance(other, Affine):
            return NotImplemented
        a, b = self.matrix, other.matrix
        rows = []
        for i in range(3):
            row = [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            row.append(sum(a[i][k] * b[k][3] for k in range(3)) + a[i][3])
            rows.append(tuple(row))
        return Affine(tuple(rows))  # type: ignore[arg-type]

    def inverse(self) -> Affine:
        (a, b, c, tx), (d, e, f, ty), (g, h, i, tz) = self.matrix
        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        if det == 0.0:
            raise ValueError("matrix is singular")
        inv = (
            ((e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det),
            ((f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det),
            ((d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det),
        )
        t = (tx, ty, tz)
        rows = tuple((*r, -sum(r[k] * t[k] for k in range(3))) for r in inv)
        return Affine(rows)  # type: ignore[arg-type]

    def to_numpy(self):
        """The 3x4 matrix as a float64 ``numpy.ndarray`` (requires numpy)."""
        import numpy as np

        return np.array(self.matrix, dtype=np.float64)

    def to_homogeneous(self) -> list[list[float]]:
        """The 4x4 homogeneous matrix as nested lists."""
        return [list(r) for r in self.matrix] + [[0.0, 0.0, 0.0, 1.0]]
