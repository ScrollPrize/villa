"""Write Automated Fiber Volumes (.afv), the read-only fiber collections of VC3D.

Format version 1 is specified in ``volume-cartographer/docs/fiber-collections.md``.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


APPLICATION_ID = 0x56434643  # "VCFC"
FORMAT_VERSION = 1
BLOCK_SEGMENTS = 256
SCHEMA = """
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE fibers(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, family TEXT NOT NULL,
  point_count INTEGER NOT NULL CHECK(point_count >= 2), length REAL NOT NULL,
  min_x REAL NOT NULL, max_x REAL NOT NULL, min_y REAL NOT NULL, max_y REAL NOT NULL,
  min_z REAL NOT NULL, max_z REAL NOT NULL, annotation TEXT NOT NULL);
CREATE INDEX fibers_length ON fibers(length DESC, id);
CREATE TABLE blocks(
  id INTEGER PRIMARY KEY, fiber_id INTEGER NOT NULL REFERENCES fibers(id),
  first_segment INTEGER NOT NULL, points BLOB NOT NULL,
  UNIQUE(fiber_id, first_segment));
CREATE VIRTUAL TABLE block_bounds USING rtree(id, min_x, max_x, min_y, max_y, min_z, max_z);
"""
_FLOAT32_MAX = float(np.finfo(np.float32).max)


@dataclass(frozen=True)
class Fiber:
    """A polyline in native L0 XYZ voxels, with its VC3D fiber annotation."""

    name: str
    family: str
    points: np.ndarray
    annotation: Mapping[str, Any]


def _dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def _bounds(points: np.ndarray) -> tuple[float, ...]:
    lower, upper = points.min(axis=0), points.max(axis=0)
    return tuple(float(v) for axis in range(3) for v in (lower[axis], upper[axis]))


def _validated(fiber: Fiber) -> np.ndarray:
    points = np.asarray(fiber.points, dtype="<f8")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError(f"Fiber {fiber.name!r} needs at least two XYZ points")
    # The RTree stores single-precision bounds.
    if not np.isfinite(points).all() or np.abs(points).max() > _FLOAT32_MAX:
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


def write_afv(
    path: str | os.PathLike,
    fibers: Iterable[Fiber],
    *,
    frame: Mapping[str, Any],
    root: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``fibers`` to a new .afv file, IDs 1, 2, ... in iteration order.

    The file appears at ``path`` only once it is complete and checked, and an
    existing file is never replaced. ``metadata`` adds keys to the reserved
    ones (``complete``, ``uuid``, ``frame``, ``root``, ``fiber_count``,
    ``point_count``). Returns the UUID and counts of the written file.
    """
    destination = Path(path)
    if destination.suffix.lower() != ".afv":
        raise ValueError("Automated Fiber Volumes must use the .afv extension")
    if not frame.get("vc_open_data_coordinate_space"):
        raise ValueError("frame needs vc_open_data_coordinate_space")
    if frame.get("vc_open_data_source_coordinate_level", 0) != 0 or frame.get("vc_open_data_source_coordinate_scale_factor", 1) != 1:
        raise ValueError("Format version 1 stores native L0 coordinates only")
    extra = dict(metadata or {})
    if extra.keys() & {"complete", "uuid", "frame", "root", "fiber_count", "point_count"}:
        raise ValueError("metadata cannot replace reserved keys")
    if destination.exists():
        raise FileExistsError(destination)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    db = None
    try:
        db = sqlite3.connect(temporary)
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(SCHEMA)
        db.execute(f"PRAGMA application_id={APPLICATION_ID}")
        db.execute(f"PRAGMA user_version={FORMAT_VERSION}")
        fiber_count = point_count = block_count = 0
        for fiber in fibers:
            points = _validated(fiber)
            fiber_count += 1
            point_count += len(points)
            length = math.fsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
            db.execute(
                "INSERT INTO fibers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (fiber_count, fiber.name, fiber.family, len(points), length, *_bounds(points), _dumps(dict(fiber.annotation))),
            )
            for first in range(0, len(points) - 1, BLOCK_SEGMENTS):
                block = points[first : first + BLOCK_SEGMENTS + 1]
                block_count += 1
                db.execute("INSERT INTO blocks VALUES(?,?,?,?)", (block_count, fiber_count, first, block.tobytes()))
                db.execute("INSERT INTO block_bounds VALUES(?,?,?,?,?,?,?)", (block_count, *_bounds(block)))
        collection_uuid = str(uuid.uuid4())
        reserved = {
            "uuid": collection_uuid,
            "frame": dict(frame),
            "root": dict(root),
            "fiber_count": fiber_count,
            "point_count": point_count,
        }
        db.executemany("INSERT INTO metadata VALUES(?,?)", [(k, _dumps(v)) for k, v in {**reserved, **extra}.items()])
        # Written last: readers refuse files without it.
        db.execute("INSERT INTO metadata VALUES('complete', 'true')")
        db.commit()
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok" or db.execute("PRAGMA foreign_key_check").fetchone():
            raise RuntimeError("The written Automated Fiber Volume failed its integrity check")
        db.close()
        db = None
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        _publish(temporary, destination)
        return {"uuid": collection_uuid, "fiber_count": fiber_count, "point_count": point_count, "block_count": block_count}
    finally:
        if db is not None:
            db.close()
        temporary.unlink(missing_ok=True)
