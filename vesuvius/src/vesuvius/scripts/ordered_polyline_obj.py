"""Strict reader for grouped ordered-polyline Wavefront OBJ files."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ObjComment:
    """One comment without its leading marker."""

    line_number: int
    text: str


@dataclass(frozen=True)
class OrderedPolylineObjGroup:
    """One named OBJ container reconstructed as one ordered path."""

    name: str
    points_xyz: np.ndarray
    comments: tuple[ObjComment, ...]
    line_number: int
    singleton: bool


@dataclass(frozen=True)
class OrderedPolylineObj:
    """Strict grouped-line OBJ contents."""

    preamble_comments: tuple[ObjComment, ...]
    groups: tuple[OrderedPolylineObjGroup, ...]


@dataclass
class _PendingGroup:
    name: str
    line_number: int
    comments: list[ObjComment]
    vertex_indices: list[int]
    vertices_xyz: list[tuple[float, float, float]]
    line_records: list[tuple[int, list[int]]]
    point_records: list[tuple[int, int]]


def read_ordered_polyline_obj(
    path: str | Path,
    *,
    container_records: Iterable[str],
    allow_singletons: bool,
    require_segment_lines: bool,
) -> OrderedPolylineObj:
    """Read strict global-indexed OBJ containers as ordered polylines.

    Only comments, configured container records, XYZ vertices, line records,
    and optional singleton point records are accepted. Every container must own
    exactly the vertices used by one nonbranching path.
    """

    obj_path = Path(path).expanduser()
    containers = frozenset(container_records)
    if not containers or not containers <= {"g", "o"}:
        raise ValueError("container_records must be a nonempty subset of {'g', 'o'}")

    preamble_comments: list[ObjComment] = []
    groups: list[OrderedPolylineObjGroup] = []
    group_names: set[str] = set()
    current: _PendingGroup | None = None
    next_vertex_index = 1

    def fail(line_number: int, message: str) -> ValueError:
        return ValueError(f"{obj_path}:{line_number}: {message}")

    def checked_indices(
        fields: list[str], line_number: int, record: str
    ) -> list[int]:
        try:
            indices = [int(value) for value in fields]
        except ValueError as exc:
            raise fail(line_number, f"{record} indices must be integers") from exc
        if any(index <= 0 for index in indices):
            raise fail(line_number, f"{record} indices must be positive")
        assert current is not None
        owned = set(current.vertex_indices)
        invalid = next((index for index in indices if index not in owned), None)
        if invalid is not None:
            raise fail(
                line_number,
                f"{record} index {invalid} does not reference a vertex in "
                f"container {current.name!r}",
            )
        return indices

    def finish_group(line_number: int) -> None:
        nonlocal current
        if current is None:
            return
        if not current.vertex_indices:
            raise fail(
                line_number,
                f"container {current.name!r} has no vertices",
            )
        if current.line_records and current.point_records:
            raise fail(
                line_number,
                f"container {current.name!r} mixes line and point records",
            )

        singleton = bool(current.point_records)
        if singleton:
            if not allow_singletons:
                raise fail(
                    current.point_records[0][0],
                    f"container {current.name!r} uses an unsupported point record",
                )
            if len(current.point_records) != 1 or len(current.vertex_indices) != 1:
                raise fail(
                    line_number,
                    f"container {current.name!r} is not one singleton point",
                )
            if current.point_records[0][1] != current.vertex_indices[0]:
                raise fail(
                    current.point_records[0][0],
                    f"container {current.name!r} point does not reference its vertex",
                )
            ordered_indices = [current.point_records[0][1]]
        else:
            if not current.line_records:
                raise fail(
                    line_number,
                    f"container {current.name!r} has no line record",
                )
            ordered_indices: list[int] = []
            for record_line, indices in current.line_records:
                if require_segment_lines and len(indices) != 2:
                    raise fail(
                        record_line,
                        "line record must reference exactly two vertices",
                    )
                if not ordered_indices:
                    ordered_indices.extend(indices)
                elif ordered_indices[-1] == indices[0]:
                    ordered_indices.extend(indices[1:])
                else:
                    raise fail(
                        record_line,
                        f"container {current.name!r} line records do not form "
                        "one ordered path",
                    )
            if len(set(ordered_indices)) != len(ordered_indices):
                raise fail(
                    line_number,
                    f"container {current.name!r} path branches or cycles",
                )
            if set(ordered_indices) != set(current.vertex_indices):
                raise fail(
                    line_number,
                    f"container {current.name!r} has unused or disconnected vertices",
                )

        local_by_index = dict(zip(current.vertex_indices, current.vertices_xyz))
        points_xyz = np.asarray(
            [local_by_index[index] for index in ordered_indices],
            dtype=np.float64,
        )
        groups.append(
            OrderedPolylineObjGroup(
                name=current.name,
                points_xyz=points_xyz,
                comments=tuple(current.comments),
                line_number=current.line_number,
                singleton=singleton,
            )
        )
        current = None

    try:
        with obj_path.open() as stream:
            final_line_number = 1
            for line_number, raw_line in enumerate(stream, start=1):
                final_line_number = line_number + 1
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    comment = ObjComment(line_number, line[1:].strip())
                    if current is None:
                        preamble_comments.append(comment)
                    else:
                        current.comments.append(comment)
                    continue

                fields = line.split()
                record = fields[0]
                if record in containers:
                    if len(fields) != 2:
                        raise fail(
                            line_number,
                            f"{record} record must contain exactly one name",
                        )
                    finish_group(line_number)
                    name = fields[1]
                    if name in group_names:
                        raise fail(line_number, f"duplicate container {name!r}")
                    group_names.add(name)
                    current = _PendingGroup(
                        name=name,
                        line_number=line_number,
                        comments=[],
                        vertex_indices=[],
                        vertices_xyz=[],
                        line_records=[],
                        point_records=[],
                    )
                elif record == "v":
                    if current is None:
                        raise fail(line_number, "vertex appears before the first container")
                    if len(fields) != 4:
                        raise fail(line_number, "vertex record must contain X Y Z")
                    try:
                        xyz = tuple(float(value) for value in fields[1:])
                    except ValueError as exc:
                        raise fail(
                            line_number, "vertex coordinates must be numeric"
                        ) from exc
                    if not np.isfinite(xyz).all():
                        raise fail(line_number, "vertex coordinates must be finite")
                    current.vertex_indices.append(next_vertex_index)
                    current.vertices_xyz.append(xyz)
                    next_vertex_index += 1
                elif record == "l":
                    if current is None:
                        raise fail(line_number, "line appears before the first container")
                    if len(fields) < 3:
                        raise fail(
                            line_number,
                            "line record must reference at least two vertices",
                        )
                    current.line_records.append(
                        (line_number, checked_indices(fields[1:], line_number, "line"))
                    )
                elif record == "p":
                    if current is None:
                        raise fail(line_number, "point appears before the first container")
                    if len(fields) != 2:
                        raise fail(
                            line_number,
                            "point record must reference exactly one vertex",
                        )
                    point_index = checked_indices(fields[1:], line_number, "point")[0]
                    current.point_records.append((line_number, point_index))
                else:
                    raise fail(line_number, f"unsupported OBJ record {record!r}")
            finish_group(final_line_number)
    except OSError as exc:
        raise ValueError(f"cannot read line OBJ {obj_path}: {exc}") from exc

    return OrderedPolylineObj(
        preamble_comments=tuple(preamble_comments),
        groups=tuple(groups),
    )
