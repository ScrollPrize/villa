"""Evaluate a saved winding graph against independent TIFXYZ meshes.

The evaluation is read-only: it opens the existing cache, attaches sampled
holdout points to its patches, and compares the resulting expected constraints
with the graph's already-stored lifted winding relations. Both the chosen
spanning-tree representative and the residual modulo the component's holonomy
period are reported. It never adds patches, registers sources, replays tracks,
or saves the graph.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Iterable

import numpy as np
import tifffile

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from spiral_graph import InputRole, SpiralThetaProvider, WindingGraph


WINDING_RE = re.compile(r"^w(\d+)_")


def mesh_points(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(
        np.asarray(tifffile.memmap(path / f"{axis}.tif", mode="r"))
        for axis in "xyz"
    )
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError(f"coordinate shapes differ in {path}")
    valid = np.logical_and.reduce(
        [np.isfinite(array) & (array >= 0.0) for array in arrays]
    )
    return arrays[0], arrays[1], arrays[2], valid


def evenly_spaced_indices(size: int, count: int) -> np.ndarray:
    if size <= 0 or count <= 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.linspace(0, size - 1, min(size, count), dtype=np.int64))


def add_collection(
    collections: dict[str, dict],
    catalog: dict[str, dict],
    name: str,
    xyz: Iterable[tuple[float, float, float]],
    windings: Iterable[int],
) -> None:
    points = {
        str(index): {"p": [float(x), float(y), float(z)], "wind_a": int(winding)}
        for index, ((x, y, z), winding) in enumerate(zip(xyz, windings, strict=True))
    }
    if len(points) < 2:
        return
    collection_id = str(len(collections))
    collections[collection_id] = {"name": name, "points": points}
    catalog[name] = {"points": len(points)}


def add_mesh_lines(
    collections: dict[str, dict],
    catalog: dict[str, dict],
    path: Path,
    winding: int,
    line_count: int,
    vertex_step: int,
    kind: str,
) -> None:
    x, y, z, valid = mesh_points(path)
    rows, columns = valid.shape
    for axis, line_indices, cross_size in (
        ("row", evenly_spaced_indices(rows, line_count), columns),
        ("column", evenly_spaced_indices(columns, line_count), rows),
    ):
        for line in line_indices:
            mask = valid[line, :] if axis == "row" else valid[:, line]
            padded = np.r_[False, mask, False]
            starts = np.flatnonzero(~padded[:-1] & padded[1:])
            stops = np.flatnonzero(padded[:-1] & ~padded[1:])
            for segment, (start, stop) in enumerate(zip(starts, stops, strict=True)):
                indices = np.arange(start, stop, vertex_step, dtype=np.int64)
                if len(indices) < 2:
                    continue
                if indices[-1] != stop - 1:
                    indices = np.r_[indices, stop - 1]
                if axis == "row":
                    xyz = zip(x[line, indices], y[line, indices], z[line, indices])
                else:
                    xyz = zip(x[indices, line], y[indices, line], z[indices, line])
                name = f"{kind}|{path.name}|{axis}{int(line)}|segment{segment}"
                add_collection(
                    collections, catalog, name, xyz,
                    np.full(len(indices), winding, dtype=np.int64),
                )


def valid_xyz(path: Path) -> np.ndarray:
    x, y, z, valid = mesh_points(path)
    return np.column_stack((x[valid], y[valid], z[valid]))


def add_named_winding_pairs(
    collections: dict[str, dict],
    catalog: dict[str, dict],
    meshes: list[tuple[int, Path]],
    samples_per_pair: int,
    gaps: list[int],
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    points = {winding: valid_xyz(path) for winding, path in meshes}
    paths = {winding: path for winding, path in meshes}
    windings = sorted(points)
    available = set(windings)
    for gap in gaps:
        for left in windings:
            right = left + gap
            if right not in available:
                continue
            left_points = points[left]
            right_points = points[right]
            if not len(left_points) or not len(right_points):
                continue
            left_samples = rng.integers(len(left_points), size=samples_per_pair)
            right_samples = rng.integers(len(right_points), size=samples_per_pair)
            for sample, (a, b) in enumerate(zip(left_samples, right_samples, strict=True)):
                name = (
                    f"named_gap_{gap}|{paths[left].name}|{paths[right].name}"
                    f"|sample{sample}"
                )
                add_collection(
                    collections,
                    catalog,
                    name,
                    [left_points[a], right_points[b]],
                    [left, right],
                )


def quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.int64)
    return {
        "min": int(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
    }


def lifted_residual(value: int, period: int) -> int:
    """Return the smallest residual in an integer-sheet equivalence class."""
    if period == 0:
        return value
    remainder = value % period
    if 2 * remainder > period:
        remainder -= period
    return remainder


def summarize(rows: list[dict], catalog: dict[str, dict], group_key) -> dict:
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = group_key(row["collection"])
        if key is not None:
            by_kind[key].append(row)
    collection_kinds = Counter(
        key for name in catalog if (key := group_key(name)) is not None
    )

    output = {}
    for kind in sorted(set(collection_kinds) | set(by_kind)):
        group = by_kind[kind]
        cross_patch = [row for row in group if row["from_patch"] != row["to_patch"]]
        connected = [row for row in group if row["residual"] is not None]
        absolute = [abs(row["residual"]) for row in connected]
        gauge_absolute = [abs(row["gauge_residual"]) for row in connected]
        covered_collections = len({row["collection"] for row in group})
        output[kind] = {
            "collections": collection_kinds[kind],
            "collections_with_attachments": covered_collections,
            "constraints": len(group),
            "cross_patch_constraints": len(cross_patch),
            "same_patch_constraints": len(group) - len(cross_patch),
            "connected_constraints": len(connected),
            "disconnected_constraints": sum(
                row["residual"] is None for row in group
            ),
            "unique_constraints": sum(
                row["holonomy_period"] == 0 for row in connected
            ),
            "sheet_ambiguous_constraints": sum(
                row["holonomy_period"] > 0 for row in connected
            ),
            "exact": sum(value == 0 for value in absolute),
            "within_one": sum(value <= 1 for value in absolute),
            "exact_rate": (
                sum(value == 0 for value in absolute) / len(absolute)
                if absolute else None
            ),
            "mean_absolute_error": float(np.mean(absolute)) if absolute else None,
            "absolute_residual": quantiles(absolute),
            "gauge_absolute_residual": quantiles(gauge_absolute),
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--umbilicus", type=Path, required=True)
    parser.add_argument("--cut-windings", type=Path, required=True)
    parser.add_argument(
        "--reference-mesh", type=Path, action="append", default=[],
        help="large continuous TIFXYZ mesh; may be repeated",
    )
    parser.add_argument("--samples-per-pair", type=int, default=64)
    parser.add_argument(
        "--winding-gaps", default="1,2,4,8,16,32",
        help="comma-separated differences tested between named wNN meshes",
    )
    parser.add_argument("--line-count", type=int, default=8)
    parser.add_argument("--vertex-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--keep-point-collections", type=Path,
        help="retain the generated holdout PCL JSON instead of using a temp file",
    )
    args = parser.parse_args()
    if args.samples_per_pair <= 0 or args.line_count < 0 or args.vertex_step <= 0:
        parser.error("sample/line counts and vertex step must be positive")
    return args


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    winding_meshes = []
    for path in sorted(args.cut_windings.iterdir()):
        match = WINDING_RE.match(path.name)
        if path.is_dir() and match:
            winding_meshes.append((int(match.group(1)), path))
    if len(winding_meshes) < 2:
        raise SystemExit(f"found fewer than two wNN meshes in {args.cut_windings}")

    collections: dict[str, dict] = {}
    catalog: dict[str, dict] = {}
    gaps = sorted({int(value) for value in args.winding_gaps.split(",") if value})
    add_named_winding_pairs(
        collections, catalog, winding_meshes, args.samples_per_pair, gaps, args.seed,
    )
    for winding, path in winding_meshes:
        add_mesh_lines(
            collections, catalog, path, winding,
            args.line_count, args.vertex_step, "cut_mesh",
        )
    for path in args.reference_mesh:
        add_mesh_lines(
            collections, catalog, path, 0,
            args.line_count, args.vertex_step, "reference_mesh",
        )

    document = {
        "vc_pointcollections_json_version": "1",
        "collections": collections,
    }
    temporary = None
    if args.keep_point_collections:
        pcl_path = args.keep_point_collections
        pcl_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="winding-holdout-", delete=False,
        )
        pcl_path = Path(temporary.name)
    try:
        if temporary is not None:
            json.dump(document, temporary, separators=(",", ":"))
            temporary.close()
        else:
            pcl_path.write_text(json.dumps(document, separators=(",", ":")))

        provider = SpiralThetaProvider(
            args.checkpoint, umbilicus=args.umbilicus, device=args.device,
        )
        graph = WindingGraph.open(args.cache, provider)
        constraints = graph.inspect_point_collections([pcl_path], InputRole.RELATIVE)

        rows = []
        for constraint in constraints:
            item = constraint.provenance.item
            collection = item.rsplit(":", 1)[0]
            kind = collection.split("|", 1)[0]
            from_patch = graph.node_name(constraint.from_node)
            to_patch = graph.node_name(constraint.to_node)
            lifted = graph.lifted_relative_winding(from_patch, to_patch)
            predicted = None if lifted is None else lifted.representative
            period = None if lifted is None else lifted.period
            gauge_residual = (
                None if predicted is None else predicted - constraint.delta
            )
            rows.append(
                {
                    "kind": kind,
                    "collection": collection,
                    "from_patch": from_patch,
                    "to_patch": to_patch,
                    "expected": constraint.delta,
                    "predicted": predicted,
                    "holonomy_period": period,
                    "gauge_residual": gauge_residual,
                    "residual": (
                        None if gauge_residual is None
                        else lifted_residual(gauge_residual, period)
                    ),
                }
            )

        stats = graph.stats()
        report = {
            "read_only": True,
            "seconds": time.perf_counter() - started,
            "cache": str(args.cache),
            "cache_stats": {
                "patches": stats.patch_count,
                "constraints": stats.constraint_count,
                "components": stats.component_count,
                "holonomies": stats.holonomy_count,
            },
            "winding_meshes": len(winding_meshes),
            "reference_meshes": [str(path) for path in args.reference_mesh],
            "generated_collections": len(collections),
            "generated_points": sum(value["points"] for value in catalog.values()),
            "summary": summarize(
                rows, catalog, lambda name: name.split("|", 1)[0],
            ),
            "mesh_summary": summarize(
                rows,
                catalog,
                lambda name: (
                    "|".join(name.split("|")[:2])
                    if name.split("|", 1)[0] in {"cut_mesh", "reference_mesh"}
                    else None
                ),
            ),
            "residual_histogram": dict(
                sorted(Counter(row["residual"] for row in rows if row["residual"] is not None).items())
            ),
            "worst": sorted(
                (row for row in rows if row["residual"] is not None),
                key=lambda row: abs(row["residual"]), reverse=True,
            )[:50],
            "worst_gauge_representatives": sorted(
                (row for row in rows if row["gauge_residual"] is not None),
                key=lambda row: abs(row["gauge_residual"]), reverse=True,
            )[:50],
        }
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")
    finally:
        if temporary is not None:
            pcl_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
