"""Publish generated fibers as a native vc_object_store database."""
from __future__ import annotations

import getpass
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class Fiber:
    """A polyline in native L0 XYZ voxels, with its VC3D fiber annotation."""
    name: str
    family: str
    points: np.ndarray
    annotation: Mapping[str, Any]


def _validated(fiber: Fiber) -> np.ndarray:
    points = np.ascontiguousarray(fiber.points, dtype="<f8")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError(f"Fiber {fiber.name!r} needs at least two XYZ points")
    if not np.isfinite(points).all() or np.abs(points).max() > float(np.finfo(np.float32).max):
        raise ValueError(f"Fiber {fiber.name!r} has non-finite coordinates")
    if fiber.family not in ("H", "V", ""):
        raise ValueError(f"Fiber {fiber.name!r} has invalid family {fiber.family!r}")
    if "line_points" in fiber.annotation:
        raise ValueError("Fiber annotations are stored without line_points")
    return points


def _publish(temporary: Path, destination: Path) -> None:
    """Move a finished file into place without replacing an existing file."""
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise
    except OSError as error:
        # Some file systems have no hard links. os.rename never replaces an
        # existing file on Windows; elsewhere it could replace a concurrent result.
        if os.name != "nt":
            raise OSError(f"{destination.parent} does not support hard links, needed to create {destination.name} safely") from error
        os.rename(temporary, destination)
    else:
        temporary.unlink()


def write_store(
    path: str | os.PathLike,
    fibers: Iterable[Fiber],
    *,
    coordinates: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a new .vcobj, publishing only after all geometry is committed.

    Geometry and summaries use the native store API. The destination is never
    replaced. Each fiber carries the volume binding and generator provenance.
    """
    from vc_object_store import ObjectStore, Fiber as StoredFiber

    destination = Path(path)
    if destination.suffix.lower() != ".vcobj":
        raise ValueError("Generated fibers must use the .vcobj extension")
    if not coordinates.get("volume_uuid") or coordinates.get("scale_factor") != 1:
        raise ValueError("coordinates need volume_uuid and native scale_factor 1")
    # Validate provenance before creating a database (including NaNs).
    extra = dict(metadata or {})
    json.dumps(extra, allow_nan=False)
    if extra.keys() & {"coordinates", "source"}:
        raise ValueError("metadata cannot replace reserved keys")
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as directory:
        temporary = Path(directory) / destination.name
        with ObjectStore.create(str(temporary), author=getpass.getuser()) as store:
            fiber_count = point_count = 0
            with store.write() as tx:
                for fiber in fibers:
                    points = _validated(fiber)
                    source = dict(fiber.annotation)
                    controls = np.asarray(source.pop("control_points", [points[0], points[-1]]), dtype="<f8")
                    if controls.ndim != 2 or controls.shape[1] != 3 or not np.isfinite(controls).all():
                        raise ValueError(f"Fiber {fiber.name!r} has invalid control points")
                    source.update(type="vc3d_fiber", version=4)
                    json.dumps(source, allow_nan=False)
                    geometry = StoredFiber.from_arrays(np.ascontiguousarray(controls), points)
                    uid = tx.upsert_object(dict(
                        uuid=str(uuid.uuid4()), kind="fiber", name=fiber.name, format_version=4,
                        fiber=dict(family=fiber.family, manual_family=bool(fiber.family), optimization_mode="lasagna",
                                   control_point_count=len(controls), line_point_count=len(points), length=geometry.length()),
                        meta={**extra, "coordinates": dict(coordinates), "source": source},
                    ))
                    tx.put_geometry(uid, geometry)
                    fiber_count += 1
                    point_count += len(points)
            store_uuid = store.uuid
            store.checkpoint()
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        _publish(temporary, destination)
    return {"uuid": store_uuid, "fiber_count": fiber_count, "point_count": point_count}
