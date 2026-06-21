from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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
    fiber_path = Path(path) if path is not None else None
    if not isinstance(obj, dict):
        raise ValueError(f"vc3d_fiber JSON must be an object, got {type(obj).__name__}")
    if obj.get("type", "vc3d_fiber") != "vc3d_fiber":
        raise ValueError(
            f"vc3d_fiber type must be 'vc3d_fiber', got {obj.get('type')!r}"
        )

    version = int(obj.get("version", 1))
    if version != 1:
        raise ValueError(f"only vc3d_fiber version 1 is supported, got {version}")

    line_points = _parse_points(
        obj.get("line_points"), key="line_points", path=fiber_path, min_count=2
    )
    control_points = _parse_points(
        obj.get("control_points"),
        key="control_points",
        path=fiber_path,
        min_count=1,
    )
    generation = int(obj.get("generation", 1))

    metadata = {
        key: value
        for key, value in obj.items()
        if key not in {"type", "version", "line_points", "control_points", "generation"}
    }
    return Vc3dFiber(
        path=fiber_path,
        version=version,
        line_points_xyz=line_points,
        control_points_xyz=control_points,
        generation=generation,
        metadata=metadata,
    )


def load_vc3d_fiber(path: str | Path) -> Vc3dFiber:
    fiber_path = Path(path)
    with fiber_path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    return parse_vc3d_fiber(obj, path=fiber_path)
