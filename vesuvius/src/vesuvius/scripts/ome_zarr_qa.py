#!/usr/bin/env python3
"""Validate the shape/metadata invariants of an OME-Zarr multiscale pyramid.

This is a CPU-only, network-free QA pass over a *built* OME-Zarr group. It reads
**only** zarr metadata (``.zarray`` / ``.zattrs`` for zarr v2, ``zarr.json`` for
zarr v3) plus small chunk-directory listings — it never loads pixel data — so it
is cheap enough to run in CI on every produced volume.

It exists to catch a family of real pyramid-production bugs seen in the field:

* A downsampled level that is one voxel short on an odd axis, because the builder
  used banker's rounding (``round``) instead of ceil-division. VC3D expects every
  level ``== ceil(n / 2**level)`` per axis (its ``ceilDivPow2``); a short level
  makes the viewer read past the array. (villa #2135, #2092)
* ``.zattrs`` multiscales missing physical scale metadata (axis units +
  ``coordinateTransformations`` scale), so viewers cannot draw a scale bar, or
  carrying scale ratios that are not ~2x between consecutive levels. (villa #497)
* Level-0 chunk objects whose on-disk size is a large multiple (8x / 64x) of the
  declared ``chunks * dtype`` size — a sign the chunk grid was mis-declared.
  (villa #1965)
* A sibling ``metadata.json`` that is missing its ``scan`` key (volume refuses to
  open) or whose ``z_direction_is_top_to_bottom`` is null/absent. (villa #2078, #2106)
* Gaps in the present level directories vs. the declared level count.

Checks (each yields a structured :class:`CheckResult` with name / status / message /
details; status is one of PASS / FAIL / WARN / SKIP):

  1. ``pyramid_level_shapes``  — level L shape[axis] == ceil(level0[axis] / 2**L).
  2. ``ome_scale_metadata``    — multiscales datasets, scale ratios ~2x, axis units.
  3. ``chunk_object_sizes``    — sampled level-0 chunk files vs declared chunk bytes.
  4. ``metadata_json_keys``    — sibling metadata.json has scan/voxelsize; z-direction set.
  5. ``level_z_consistency``   — declared level count vs present level dirs; no gaps.

Usage::

    # human-readable table; exit nonzero if any check FAILs
    python -m vesuvius.scripts.ome_zarr_qa /path/to/volume.zarr

    # limit to the first N levels, emit machine-readable JSON
    python -m vesuvius.scripts.ome_zarr_qa /path/to/volume.zarr --levels 4 --json

    # CI gate: treat WARN as failure too
    python -m vesuvius.scripts.ome_zarr_qa /path/to/volume.zarr --strict

As a library::

    from vesuvius.scripts.ome_zarr_qa import run_all_checks
    results = run_all_checks("/path/to/volume.zarr")
    failed = [r for r in results if r.status == "FAIL"]

Dependencies are stdlib + numpy only (numpy is used solely for ``dtype.itemsize``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Status constants and the result record
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"


@dataclass
class CheckResult:
    """One QA check outcome. ``details`` must be JSON-serialisable."""

    name: str
    status: str
    message: str
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class ArrayMeta:
    """Normalised view of a single level's array metadata (v2 or v3)."""

    level: int
    path: str
    shape: Tuple[int, ...]
    chunks: Tuple[int, ...]
    dtype: str
    itemsize: int
    dimension_separator: str
    zarr_format: int


# ---------------------------------------------------------------------------
# Metadata readers (no pixel data, ever)
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[dict]:
    """Return parsed JSON at ``path`` or None if it does not exist.

    Raises on a present-but-corrupt file so callers can surface it.
    """
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _itemsize(dtype_str: str) -> int:
    """Byte width of a (v2 or v3) dtype string, via numpy."""
    return int(np.dtype(dtype_str).itemsize)


def read_array_meta(level: int, level_path: Path) -> ArrayMeta:
    """Read one level's array metadata, supporting zarr v2 and v3.

    Raises ValueError if neither ``.zarray`` nor an array ``zarr.json`` is found.
    """
    zarray = _read_json(level_path / ".zarray")
    if zarray is not None:
        shape = tuple(int(v) for v in zarray["shape"])
        chunks = tuple(int(v) for v in zarray["chunks"])
        dtype = str(zarray["dtype"])
        sep = zarray.get("dimension_separator", ".")
        return ArrayMeta(
            level=level,
            path=str(level_path),
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            itemsize=_itemsize(dtype),
            dimension_separator=sep,
            zarr_format=2,
        )

    zarr_json = _read_json(level_path / "zarr.json")
    if zarr_json is not None and zarr_json.get("node_type") == "array":
        shape = tuple(int(v) for v in zarr_json["shape"])
        grid = zarr_json["chunk_grid"]["configuration"]["chunk_shape"]
        chunks = tuple(int(v) for v in grid)
        dtype = str(zarr_json["data_type"])
        sep = "/"
        cke = zarr_json.get("chunk_key_encoding", {})
        sep = cke.get("configuration", {}).get("separator", "/")
        return ArrayMeta(
            level=level,
            path=str(level_path),
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            itemsize=_itemsize(dtype),
            dimension_separator=sep,
            zarr_format=3,
        )

    raise ValueError(f"no .zarray or array zarr.json at {level_path}")


def discover_levels(group_path: Path, max_levels: Optional[int] = None) -> List[int]:
    """Return the sorted integer level names present under ``group_path``.

    A directory counts as a level if its name parses as an int and it holds a
    ``.zarray`` or ``zarr.json``. ``max_levels`` caps how many are considered.
    """
    levels: List[int] = []
    if not group_path.is_dir():
        return levels
    for child in group_path.iterdir():
        if not child.is_dir():
            continue
        try:
            lvl = int(child.name)
        except ValueError:
            continue
        if (child / ".zarray").exists() or (child / "zarr.json").exists():
            levels.append(lvl)
    levels.sort()
    if max_levels is not None:
        levels = [l for l in levels if l < max_levels]
    return levels


def read_group_attrs(group_path: Path) -> dict:
    """Return the group's user attributes (``.zattrs`` for v2, ``attributes`` in
    ``zarr.json`` for v3). Empty dict if none."""
    zattrs = _read_json(group_path / ".zattrs")
    if zattrs is not None:
        return zattrs
    zarr_json = _read_json(group_path / "zarr.json")
    if zarr_json is not None and zarr_json.get("node_type") == "group":
        return zarr_json.get("attributes", {}) or {}
    return {}


def find_metadata_json(group_path: Path) -> Optional[Path]:
    """Locate a sibling ``metadata.json`` (inside the group, else next to it)."""
    inside = group_path / "metadata.json"
    if inside.exists():
        return inside
    sibling = group_path.parent / "metadata.json"
    if sibling.exists():
        return sibling
    return None


def ceil_div_pow2(n: int, level: int) -> int:
    """ceil(n / 2**level) — matches VC3D's ceilDivPow2."""
    denom = 1 << level
    return (n + denom - 1) // denom


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_pyramid_level_shapes(metas: Sequence[ArrayMeta]) -> CheckResult:
    """Level L must equal ceil(level0 / 2**L) on every downsampled axis.

    An axis whose size is identical in *every* level is treated as
    non-downsampled (e.g. a channel axis) and is expected to stay constant.
    """
    name = "pyramid_level_shapes"
    if len(metas) < 2:
        return CheckResult(
            name, SKIP, "fewer than two levels present; nothing to cross-check",
            {"levels": [m.level for m in metas]},
        )

    level0 = metas[0]
    ndim = len(level0.shape)
    for m in metas:
        if len(m.shape) != ndim:
            return CheckResult(
                name, FAIL,
                f"level {m.level} has {len(m.shape)} dims; level 0 has {ndim}",
                {"level": m.level, "shape": list(m.shape), "level0_shape": list(level0.shape)},
            )

    # An axis is "downsampled" unless every level shares level 0's size there.
    downsampled = [
        any(m.shape[ax] != level0.shape[ax] for m in metas)
        for ax in range(ndim)
    ]

    violations = []
    for m in metas:
        for ax in range(ndim):
            if downsampled[ax]:
                expected = ceil_div_pow2(level0.shape[ax], m.level)
            else:
                expected = level0.shape[ax]
            if m.shape[ax] != expected:
                violations.append(
                    {
                        "level": m.level,
                        "axis": ax,
                        "expected": expected,
                        "actual": m.shape[ax],
                        "level0_axis": level0.shape[ax],
                    }
                )

    if violations:
        first = violations[0]
        msg = (
            f"{len(violations)} axis/level shape mismatch(es); first: level "
            f"{first['level']} axis {first['axis']} expected {first['expected']} "
            f"(=ceil({first['level0_axis']}/2**{first['level']})) got {first['actual']}"
        )
        return CheckResult(name, FAIL, msg, {"violations": violations, "downsampled_axes": downsampled})

    return CheckResult(
        name, PASS,
        f"all {len(metas)} levels match ceil-division of level 0 {list(level0.shape)}",
        {"downsampled_axes": downsampled},
    )


def _extract_scale(dataset: dict) -> Optional[List[float]]:
    """Pull the `scale` transform out of a multiscales dataset entry, if any."""
    for ct in dataset.get("coordinateTransformations", []) or []:
        if ct.get("type") == "scale" and "scale" in ct:
            return [float(v) for v in ct["scale"]]
    return None


def check_ome_scale_metadata(group_path: Path, metas: Sequence[ArrayMeta]) -> CheckResult:
    """Validate OME multiscales: dataset count, ~2x scale ratios, axis units."""
    name = "ome_scale_metadata"
    attrs = read_group_attrs(group_path)
    multiscales = attrs.get("multiscales")
    if not multiscales:
        return CheckResult(
            name, WARN,
            "no `multiscales` metadata on the group; viewers cannot show a scale bar (#497)",
            {},
        )

    ms = multiscales[0]
    axes = ms.get("axes", [])
    datasets = ms.get("datasets", [])

    problems: List[str] = []
    warnings: List[str] = []
    details: Dict = {}

    n_levels = len(metas) if metas else len(datasets)
    if len(datasets) != n_levels:
        problems.append(
            f"multiscales lists {len(datasets)} datasets but {n_levels} levels are present"
        )

    # Axis names + units.
    axis_names = [a.get("name") for a in axes]
    details["axes"] = axes
    if any(n is None for n in axis_names):
        problems.append("one or more axes are missing a `name`")
    units = [a.get("unit") for a in axes]
    if any(u is None for u in units):
        warnings.append("one or more axes are missing a physical `unit`")

    # Scale transforms + ratios.
    scales = [_extract_scale(d) for d in datasets]
    details["scales"] = scales
    if all(s is None for s in scales):
        warnings.append(
            "no coordinateTransformations `scale` on any dataset; "
            "physical voxel size is unknown (#497)"
        )
    else:
        # Report derivable physical voxel size (level-0 scale + units).
        if scales and scales[0] is not None:
            details["voxel_size_level0"] = scales[0]
            if units:
                details["voxel_size_units"] = units

        downsampled = None
        if len(metas) >= 2:
            ndim = len(metas[0].shape)
            downsampled = [
                any(m.shape[ax] != metas[0].shape[ax] for m in metas)
                for ax in range(ndim)
            ]

        for i in range(1, len(scales)):
            prev, cur = scales[i - 1], scales[i]
            if prev is None or cur is None:
                warnings.append(f"dataset {i} or {i - 1} is missing a scale transform")
                continue
            if len(prev) != len(cur):
                problems.append(f"scale length differs between datasets {i-1} and {i}")
                continue
            for ax in range(len(cur)):
                is_down = True if downsampled is None else downsampled[ax]
                ratio = cur[ax] / prev[ax] if prev[ax] else float("nan")
                if is_down:
                    if not (math.isfinite(ratio) and abs(ratio - 2.0) <= 0.05 * 2.0):
                        problems.append(
                            f"scale ratio on axis {ax} between datasets {i-1}->{i} "
                            f"is {ratio:.4g}, expected ~2.0"
                        )
                else:
                    if not (math.isfinite(ratio) and abs(ratio - 1.0) <= 0.05):
                        warnings.append(
                            f"scale ratio on non-downsampled axis {ax} "
                            f"between datasets {i-1}->{i} is {ratio:.4g}, expected ~1.0"
                        )

    if problems:
        return CheckResult(
            name, FAIL, "; ".join(problems), {**details, "warnings": warnings}
        )
    if warnings:
        return CheckResult(name, WARN, "; ".join(warnings), details)
    return CheckResult(
        name, PASS, "multiscales metadata is consistent (datasets, ~2x scales, axis units)",
        details,
    )


def _chunk_files(level_path: Path, limit: int) -> List[Path]:
    """List up to ``limit`` chunk files under a level (sorted), skipping metadata."""
    out: List[Path] = []
    for p in sorted(level_path.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith(".") or p.name == "zarr.json":
            continue
        out.append(p)
        if len(out) >= limit:
            break
    return out


def check_chunk_object_sizes(
    metas: Sequence[ArrayMeta], sample: int = 32, warn_ratio: float = 1.25,
    fail_ratio: float = 2.0,
) -> CheckResult:
    """Sampled level-0 chunk files must not dwarf the declared chunk byte size.

    A stored chunk is at most its uncompressed ``prod(chunks) * itemsize`` plus a
    little compressor overhead. A file several times larger means the declared
    chunk grid does not match what was written (#1965). Listing + stat only — no
    pixel reads.
    """
    name = "chunk_object_sizes"
    if not metas:
        return CheckResult(name, SKIP, "no levels found", {})
    level0 = metas[0]
    level_path = Path(level0.path)

    declared_bytes = int(np.prod(level0.chunks)) * level0.itemsize
    if declared_bytes <= 0:
        return CheckResult(
            name, SKIP, "declared chunk byte size is zero; cannot compare", {}
        )

    files = _chunk_files(level_path, sample)
    if not files:
        return CheckResult(
            name, SKIP,
            f"level 0 has no chunk files on disk (all fill-value?)",
            {"declared_chunk_bytes": declared_bytes},
        )

    offenders = []
    max_ratio = 0.0
    for p in files:
        size = p.stat().st_size
        ratio = size / declared_bytes
        max_ratio = max(max_ratio, ratio)
        if ratio > warn_ratio:
            offenders.append(
                {
                    "chunk": p.relative_to(level_path).as_posix(),
                    "size_bytes": size,
                    "declared_bytes": declared_bytes,
                    "ratio": round(ratio, 3),
                }
            )

    details = {
        "declared_chunk_bytes": declared_bytes,
        "sampled": len(files),
        "max_ratio": round(max_ratio, 3),
        "offenders": offenders,
    }
    if any(o["ratio"] > fail_ratio for o in offenders):
        return CheckResult(
            name, FAIL,
            f"{len(offenders)} sampled chunk(s) exceed declared size; "
            f"max {max_ratio:.2f}x of {declared_bytes} bytes (chunk grid mis-declared? #1965)",
            details,
        )
    if offenders:
        return CheckResult(
            name, WARN,
            f"{len(offenders)} sampled chunk(s) slightly exceed declared size "
            f"(max {max_ratio:.2f}x)",
            details,
        )
    return CheckResult(
        name, PASS,
        f"{len(files)} sampled level-0 chunks are within the declared "
        f"{declared_bytes}-byte size (max {max_ratio:.2f}x)",
        details,
    )


def check_metadata_json_keys(group_path: Path) -> CheckResult:
    """Validate a sibling metadata.json: require scan/voxelsize; z-direction set."""
    name = "metadata_json_keys"
    md_path = find_metadata_json(group_path)
    if md_path is None:
        return CheckResult(
            name, SKIP, "no sibling metadata.json found", {}
        )
    try:
        md = _read_json(md_path)
    except Exception as exc:  # corrupt JSON
        return CheckResult(
            name, FAIL, f"metadata.json is not valid JSON: {exc}",
            {"path": str(md_path)},
        )

    required = ["scan", "voxelsize"]
    missing = [k for k in required if k not in md]
    details = {"path": str(md_path), "keys": sorted(md.keys())}

    warnings = []
    if "z_direction_is_top_to_bottom" not in md:
        warnings.append("`z_direction_is_top_to_bottom` is absent")
    elif md["z_direction_is_top_to_bottom"] is None:
        warnings.append("`z_direction_is_top_to_bottom` is null (#2106)")

    if missing:
        return CheckResult(
            name, FAIL,
            f"metadata.json missing required key(s): {missing} "
            f"(volume may refuse to open, #2078)",
            {**details, "missing": missing, "warnings": warnings},
        )
    if warnings:
        return CheckResult(name, WARN, "; ".join(warnings), details)
    return CheckResult(
        name, PASS, "metadata.json has scan/voxelsize and a set z-direction", details
    )


def check_level_z_consistency(
    group_path: Path, present_levels: Sequence[int]
) -> CheckResult:
    """Declared level count (from multiscales) vs present dirs; flag gaps."""
    name = "level_z_consistency"
    if not present_levels:
        return CheckResult(name, FAIL, "no level directories found under the group", {})

    details = {"present_levels": list(present_levels)}

    # Contiguity from 0.
    expected = list(range(len(present_levels)))
    gaps = sorted(set(range(present_levels[0], present_levels[-1] + 1)) - set(present_levels))
    if present_levels[0] != 0:
        return CheckResult(
            name, WARN, f"levels do not start at 0 (start={present_levels[0]})",
            {**details, "gaps": gaps},
        )
    if gaps:
        return CheckResult(
            name, FAIL, f"gaps in level directories: missing {gaps}",
            {**details, "gaps": gaps},
        )

    attrs = read_group_attrs(group_path)
    multiscales = attrs.get("multiscales")
    if multiscales:
        datasets = multiscales[0].get("datasets", [])
        details["declared_levels"] = len(datasets)
        if len(datasets) != len(present_levels):
            return CheckResult(
                name, WARN,
                f"multiscales declares {len(datasets)} levels but "
                f"{len(present_levels)} are present on disk",
                details,
            )

    return CheckResult(
        name, PASS, f"{len(present_levels)} contiguous levels present from 0", details
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all_checks(
    group_path, levels: Optional[int] = None, sample: int = 32
) -> List[CheckResult]:
    """Run every check against an OME-Zarr group directory.

    Any unexpected error inside a check is captured as a FAIL result rather than
    propagating, so a malformed store is reported, not crashed on.
    """
    group_path = Path(group_path)
    results: List[CheckResult] = []

    if not group_path.exists():
        return [CheckResult("store", FAIL, f"path does not exist: {group_path}", {})]

    present = discover_levels(group_path, max_levels=levels)
    if not present:
        results.append(
            CheckResult("store", FAIL, f"no zarr levels found under {group_path}", {})
        )

    metas: List[ArrayMeta] = []
    for lvl in present:
        try:
            metas.append(read_array_meta(lvl, group_path / str(lvl)))
        except Exception as exc:
            results.append(
                CheckResult(
                    "read_level", FAIL,
                    f"failed to read level {lvl} metadata: {exc}",
                    {"level": lvl},
                )
            )

    def guarded(fn, *args, name="check"):
        try:
            return fn(*args)
        except Exception as exc:  # never crash the whole run
            return CheckResult(name, FAIL, f"check raised: {exc}", {})

    results.append(guarded(check_pyramid_level_shapes, metas, name="pyramid_level_shapes"))
    results.append(guarded(check_ome_scale_metadata, group_path, metas, name="ome_scale_metadata"))
    results.append(
        guarded(lambda: check_chunk_object_sizes(metas, sample=sample), name="chunk_object_sizes")
    )
    results.append(guarded(check_metadata_json_keys, group_path, name="metadata_json_keys"))
    results.append(guarded(check_level_z_consistency, group_path, present, name="level_z_consistency"))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_STATUS_RANK = {FAIL: 0, WARN: 1, SKIP: 2, PASS: 3}


def _print_table(results: Sequence[CheckResult]) -> None:
    name_w = max((len(r.name) for r in results), default=4)
    name_w = max(name_w, len("CHECK"))
    print(f"{'CHECK':<{name_w}}  STATUS  MESSAGE")
    print(f"{'-' * name_w}  ------  {'-' * 40}")
    for r in sorted(results, key=lambda r: _STATUS_RANK.get(r.status, 9)):
        print(f"{r.name:<{name_w}}  {r.status:<6}  {r.message}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("group_path", help="Path to the OME-Zarr multiscale group directory.")
    parser.add_argument(
        "--levels", type=int, default=None,
        help="Only consider levels [0, N) (default: all present).",
    )
    parser.add_argument(
        "--sample", type=int, default=32,
        help="Max number of level-0 chunk files to stat for the size check.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--strict", action="store_true", help="Treat WARN as failure (exit nonzero)."
    )
    args = parser.parse_args(argv)

    results = run_all_checks(args.group_path, levels=args.levels, sample=args.sample)

    statuses = [r.status for r in results]
    summary = {s: statuses.count(s) for s in (PASS, FAIL, WARN, SKIP)}

    if args.json:
        print(
            json.dumps(
                {
                    "group_path": str(args.group_path),
                    "summary": summary,
                    "results": [r.to_dict() for r in results],
                },
                indent=2,
            )
        )
    else:
        print(f"OME-Zarr QA: {args.group_path}")
        _print_table(results)
        print(
            f"\nSummary: {summary[PASS]} PASS, {summary[FAIL]} FAIL, "
            f"{summary[WARN]} WARN, {summary[SKIP]} SKIP"
        )

    if summary[FAIL] > 0:
        return 1
    if args.strict and summary[WARN] > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
