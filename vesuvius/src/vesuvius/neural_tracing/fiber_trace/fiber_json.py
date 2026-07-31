from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vc3d_fiber_format import (
    FiberTraceSegmentMetadata,
    parse_vc3d_fiber_format,
)


@dataclass(frozen=True)
class Vc3dFiber:
    """Parsed VC3D fiber geometry.

    VC3D stores points as x, y, z. Zarr volume access uses z, y, x, so both
    orders are exposed explicitly.
    """

    path: Path | None
    version: int
    line_points_xyz: np.ndarray
    control_points_xyz: np.ndarray
    control_point_segments: tuple[FiberTraceSegmentMetadata | None, ...]
    generation: int
    metadata: dict[str, Any]

    @property
    def line_points_zyx(self) -> np.ndarray:
        return self.line_points_xyz[:, (2, 1, 0)].astype(np.float32, copy=True)

    @property
    def control_points_zyx(self) -> np.ndarray:
        return self.control_points_xyz[:, (2, 1, 0)].astype(np.float32, copy=True)


def _parse_points(
    raw: Any, *, key: str, path: Path | None, min_count: int
) -> np.ndarray:
    label = f"vc3d_fiber {key}"
    if path is not None:
        label = f"{label} in {path}"
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be a list")
    if len(raw) < min_count:
        raise ValueError(f"{label} must contain at least {min_count} point(s)")

    points = np.asarray(raw, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{label} must have shape [N, 3]")
    if not bool(np.isfinite(points).all()):
        raise ValueError(f"{label} contains non-finite values")
    return points.astype(np.float32, copy=False)


def parse_vc3d_fiber(
    obj: dict[str, Any], *, path: str | Path | None = None
) -> Vc3dFiber:
    parsed = parse_vc3d_fiber_format(obj, path=path)
    fiber_path = parsed.path
    line_points = _parse_points(
        list(parsed.line_points_xyz), key="line_points", path=fiber_path, min_count=2
    )
    control_points = _parse_points(
        list(parsed.control_points_xyz), key="control_points", path=fiber_path, min_count=1
    )
    metadata = dict(parsed.metadata)
    metadata["optimization_mode"] = parsed.optimization_mode
    return Vc3dFiber(
        path=fiber_path,
        version=parsed.version,
        line_points_xyz=line_points,
        control_points_xyz=control_points,
        control_point_segments=parsed.control_point_segments,
        generation=parsed.generation,
        metadata=metadata,
    )


def load_vc3d_fiber(path: str | Path) -> Vc3dFiber:
    fiber_path = Path(path)
    with fiber_path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    return parse_vc3d_fiber(obj, path=fiber_path)
