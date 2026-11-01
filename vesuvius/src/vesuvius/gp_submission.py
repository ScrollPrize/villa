"""Cut column meshes for a Grand Prize submission and fail closed on a bad folder.

``cut-columns`` splits a flattened TIFXYZ surface into one mesh per text
column, and ``check`` runs deterministic, inexpensive gates over the
submission's ``meshes/`` and ``images/`` folders.  Required gates follow the
Grand Prize rules (https://scrollprize.org/prizes#2027-grand-prize): every mesh
reads as tifxyz, meshes are named ``column_NN`` in sequence, and every image
is named after one mesh.  Warnings cover what the rules leave open (a numeric
stretch tolerance, overlapping cuts, meshes without an image of their own).
The checks do not read the text or judge legibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Sequence

import numpy as np

from vesuvius.tifxyz import Tifxyz, read_tifxyz, write_tifxyz


SCHEMA_VERSION = 1
COLUMN_NAME_RE = re.compile(r"^column_(\d{2,3})(?:_part(\d+))?\.tifxyz$")
PART_KINDS = ("break", "patch")


class SubmissionInputError(RuntimeError):
    """A required input is missing or invalid."""


def _gate(
    name: str,
    passed: bool,
    *,
    required: bool = True,
    observed: Any,
    threshold: Any = None,
    message: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "required": required,
        "passed": bool(passed),
        "observed": observed,
        "threshold": threshold,
        "message": message,
    }


def _read_cuts(path: Path, width: int) -> list[dict[str, Any]]:
    """Parse and validate every row of cuts.csv before anything is written."""
    if not path.is_file():
        raise SubmissionInputError(f"{path} is missing")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if not {"column", "u0", "u1"} <= set(reader.fieldnames or ()):
            raise SubmissionInputError(
                f"{path}: header must include column,u0,u1 (got {reader.fieldnames})"
            )
        rows = [(reader.line_num, row) for row in reader]
    if not rows:
        raise SubmissionInputError(f"{path} has no cuts")
    cuts, errors = [], []
    for line, row in rows:
        where = f"{path}:{line}"
        try:
            column, u0, u1 = int(row["column"]), int(row["u0"]), int(row["u1"])
        except (TypeError, ValueError):
            errors.append(f"{where}: column, u0 and u1 must be integers")
            continue
        kind = (row.get("kind") or "column").strip()
        if kind not in ("column", *PART_KINDS):
            errors.append(f"{where}: kind {kind!r} is not column, {' or '.join(PART_KINDS)}")
        if not 1 <= column <= 999:
            errors.append(f"{where}: column must be 1-999, got {column}")
        if not 0 <= u0 < u1 <= width:
            errors.append(
                f"{where}: need 0 <= u0 < u1 <= {width} (surface width), got u0={u0} u1={u1}"
            )
        cuts.append({"line": line, "column": column, "u0": u0, "u1": u1, "kind": kind})
    parts: dict[int, int] = {}
    for cut in cuts:
        name = f"column_{cut['column']:02d}"
        if cut["kind"] != "column":
            parts[cut["column"]] = parts.get(cut["column"], 0) + 1
            name += f"_part{parts[cut['column']]}"
        cut["mesh"] = name + ".tifxyz"
    lines: dict[str, list[str]] = {}
    for cut in cuts:
        lines.setdefault(cut["mesh"], []).append(str(cut.pop("line")))
    errors += [
        f"{path}: more than one cut for {name} (lines {', '.join(at)})"
        for name, at in lines.items()
        if len(at) > 1
    ]
    if errors:
        raise SubmissionInputError("\n".join(errors))
    return cuts


def _cmd_cut_columns(args: argparse.Namespace) -> int:
    """Split a flattened surface by grid-column ranges [u0, u1) into column meshes."""
    if not (args.surface / "meta.json").is_file():
        raise SubmissionInputError(f"{args.surface} is not a TIFXYZ directory")
    try:
        surface = read_tifxyz(args.surface)
    except Exception as error:  # report any unreadable surface as bad input
        raise SubmissionInputError(
            f"{args.surface} does not read as tifxyz (check meta.json, x.tif, y.tif and z.tif): "
            f"{type(error).__name__}: {error}"
        ) from None
    points = surface.get_zyxs(stored_resolution=True)
    valid = surface.valid_vertex_mask
    cuts = _read_cuts(args.cuts, points.shape[1])
    filled = np.flatnonzero(valid.any(axis=0))
    empty = [
        f"{cut['mesh']} (u {cut['u0']}-{cut['u1']})"
        for cut in cuts
        if not valid[:, cut["u0"] : cut["u1"]].any()
    ]
    if empty:
        span = f"{filled[0]}-{filled[-1] + 1}" if filled.size else "none"
        raise SubmissionInputError(
            f"no valid surface points in {', '.join(empty)}; "
            f"{args.surface.name} has valid points in u {span}"
        )
    mesh_dir = args.submission / "meshes"
    if mesh_dir.exists() and any(mesh_dir.iterdir()) and not args.overwrite:
        raise SubmissionInputError(f"{mesh_dir} is not empty; pass --overwrite to replace it")
    # Write into a temporary folder and swap it in, so a failed run leaves meshes/ as it was.
    args.submission.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".meshes-new-", dir=args.submission))
    try:
        _write_meshes(staging, args.surface, surface, points, valid, cuts)
        if mesh_dir.exists():
            old = Path(tempfile.mkdtemp(prefix=".meshes-old-", dir=args.submission))
            try:
                os.replace(mesh_dir, old / "meshes")
            except OSError:
                old.rmdir()
                raise
            try:
                os.replace(staging, mesh_dir)
            except OSError:
                try:
                    os.replace(old / "meshes", mesh_dir)
                except OSError:
                    raise SubmissionInputError(
                        f"could not swap in the new meshes; the old ones are in {old / 'meshes'}"
                    ) from None
                old.rmdir()
                raise
            shutil.rmtree(old)
        else:
            os.replace(staging, mesh_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    written = _count(len(cuts), "mesh", "meshes")
    print(f"cut-columns: {written} written to {mesh_dir}", file=sys.stderr)
    return 0


def _write_meshes(
    mesh_dir: Path,
    source: Path,
    surface: Tifxyz,
    points: np.ndarray,
    valid: np.ndarray,
    cuts: list[dict[str, Any]],
) -> None:
    for cut in cuts:
        piece = points[:, cut["u0"] : cut["u1"]]
        piece_valid = valid[:, cut["u0"] : cut["u1"]]
        write_tifxyz(
            mesh_dir / cut["mesh"],
            Tifxyz(
                _z=piece[..., 0],
                _y=piece[..., 1],
                _x=piece[..., 2],
                uuid=cut["mesh"][: -len(".tifxyz")],
                _scale=surface.get_scale_tuple(),
                bbox=tuple(
                    piece[piece_valid][:, ::-1].min(0).tolist()
                    + piece[piece_valid][:, ::-1].max(0).tolist()
                ),
                extra={
                    "kind": cut["kind"],
                    "source": str(source.resolve()),
                    "u0": cut["u0"],
                    "u1": cut["u1"],
                },
            ),
            overwrite=True,
        )


def _stretch(surface: Tifxyz) -> list[float] | None:
    """p05 and p95 of 3D grid-edge length over the grid step (1 / scale voxels)."""
    points = surface.get_zyxs(stored_resolution=True)
    valid = surface.valid_vertex_mask
    scale_y, scale_x = surface.get_scale_tuple()
    edges_u = np.linalg.norm(points[:, 1:] - points[:, :-1], axis=-1) * scale_x
    edges_v = np.linalg.norm(points[1:] - points[:-1], axis=-1) * scale_y
    ratios = np.concatenate(
        [edges_u[valid[:, 1:] & valid[:, :-1]], edges_v[valid[1:] & valid[:-1]]]
    )
    if not ratios.size:
        return None
    return [float(np.percentile(ratios, 5)), float(np.percentile(ratios, 95))]


def _check_meshes(root: Path, tolerance: float) -> list[dict[str, Any]]:
    paths = sorted(
        path for path in (root / "meshes").glob("*.tifxyz") if not path.name.startswith(".")
    )
    others = sorted(
        path.name
        for path in (root / "meshes").iterdir()
        if path.suffix != ".tifxyz" and not path.name.startswith(".")
    )
    unreadable, stretched, cuts = [], {}, {}
    for path in paths:
        try:
            surface = read_tifxyz(path)
            span = _stretch(surface)
        except Exception as error:  # any unreadable mesh fails this gate, not the run
            unreadable.append(f"{path.name}: {type(error).__name__}: {error}")
            continue
        if span is None:
            unreadable.append(f"{path.name}: no valid grid edges")
        elif not (1 - tolerance <= span[0] and span[1] <= 1 + tolerance):
            stretched[path.name] = [round(value, 3) for value in span]
        cuts[path.name] = surface.extra if isinstance(surface.extra, dict) else {}
    readable = len(paths) - len(unreadable)

    numbers, bad_names = set(), []
    for path in paths:
        match = COLUMN_NAME_RE.match(path.name)
        # column_1 or column_001 must be spelled column_01, and numbering starts at 1.
        if not match or match.group(1) != f"{int(match.group(1)):02d}" or match.group(1) == "00":
            bad_names.append(path.name)
            continue
        numbers.add(int(match.group(1)))
    gaps = sorted(set(range(1, max(numbers, default=0) + 1)) - numbers)
    # Meshes cut from the same surface must not share grid columns.
    ranges = sorted(
        (extra["source"], extra["u0"], extra["u1"], name)
        for name, extra in cuts.items()
        if isinstance(extra.get("source"), str)
        and all(type(extra.get(key)) is int for key in ("u0", "u1"))
    )
    overlaps = [
        f"{a[3]} and {b[3]} overlap"
        for i, a in enumerate(ranges)
        for b in ranges[i + 1 :]
        if a[0] == b[0] and b[1] < a[2]
    ]
    naming_problems = (
        [f"bad name {name}" for name in bad_names]
        + [f"{name} is not a tifxyz mesh" for name in others]
        + [f"column {number:02d} missing" for number in gaps]
    )
    return [
        _gate(
            "tifxyz_format",
            paths and not unreadable,
            observed={"mesh_count": len(paths), "problems": unreadable[:5]},
            message=(
                "No meshes in meshes/."
                if not paths
                else f"Unreadable or empty: {'; '.join(unreadable[:3])}."
                if unreadable
                else f"{len(paths)} meshes read as tifxyz."
            ),
        ),
        _gate(
            "mesh_isometry",
            readable > 0 and not stretched,
            required=False,
            observed={"mesh_count": readable, "failing_stretch_p05_p95": stretched},
            threshold={"stretch_p05_min": 1 - tolerance, "stretch_p95_max": 1 + tolerance},
            message=f"{readable - len(stretched)} of {readable} readable meshes are within "
            "the stretch tolerance"
            + "".join(
                f"; {n} p05-p95 {s[0]:.2f}-{s[1]:.2f}" for n, s in list(stretched.items())[:3]
            )
            + ".",
        ),
        _gate(
            "column_naming",
            paths and not naming_problems,
            observed={"problems": naming_problems},
            message=(
                "Meshes are named column_NN in sequence."
                if paths and not naming_problems
                else "No meshes in meshes/."
                if not paths
                else _listing(naming_problems)
            ),
        ),
        _gate(
            "no_overlapping_cuts",
            not overlaps,
            required=False,
            observed={"problems": overlaps},
            message=_listing(overlaps) if overlaps else "No two meshes share grid columns.",
        ),
    ]


def _check_images(root: Path) -> list[dict[str, Any]]:
    stems = {
        path.name[: -len(".tifxyz")]
        for path in (root / "meshes").glob("*.tifxyz")
        if not path.name.startswith(".")
    }
    images = sorted(
        path.name
        for path in (root / "images").glob("*")
        if path.is_file() and not path.name.startswith(".")
    )
    image_stems = [name.rsplit(".", 1)[0] for name in images]
    problems = [
        f"{stem} has {image_stems.count(stem)} images"
        for stem in sorted(set(image_stems))
        if image_stems.count(stem) > 1
    ] + [f"{name} has no mesh" for name in images if name.rsplit(".", 1)[0] not in stems]
    # The rules allow one image for a sequence of consecutive wraps, so a mesh
    # without an image only warns.
    missing = sorted(stems - set(image_stems))
    paired = bool(images) and not problems
    return [
        _gate(
            "image_names_match_meshes",
            paired,
            observed={"image_count": len(images), "problems": problems},
            message=(
                "Every image is named after one mesh."
                if paired
                else "No images in images/."
                if not images
                else _listing(problems)
            ),
        ),
        _gate(
            "every_mesh_has_image",
            not missing,
            required=False,
            observed={"meshes_without_image": missing},
            message=(
                "Every mesh has an image."
                if not missing
                else _listing([f"{stem} has no image" for stem in missing])
            ),
        ),
    ]


def check_submission(root: Path, iso_tolerance: float = 0.10) -> dict[str, Any]:
    if not (root / "meshes").is_dir():
        raise SubmissionInputError(f"{root / 'meshes'} is missing; run cut-columns first")
    gates = [*_check_meshes(root, iso_tolerance), *_check_images(root)]
    required = [gate for gate in gates if gate["required"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "submission": str(root.resolve()),
        "status": "PASS" if all(gate["passed"] for gate in required) else "FAIL",
        "summary": {
            "passed_required_gates": sum(gate["passed"] for gate in required),
            "required_gate_count": len(required),
            "warnings": sum(not gate["passed"] for gate in gates if not gate["required"]),
        },
        "gates": gates,
    }


def _count(number: int, one: str, many: str) -> str:
    return f"{number} {one if number == 1 else many}"


def _listing(problems: list[str]) -> str:
    return "; ".join(problems) + "."


def _result(gate: dict[str, Any]) -> str:
    return "PASS" if gate["passed"] else "FAIL" if gate["required"] else "WARN"


def _write_atomic(path: Path, text: str) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _cmd_check(args: argparse.Namespace) -> int:
    if not 0 < args.iso_tolerance < 1:
        raise SubmissionInputError(
            f"--iso-tolerance must be between 0 and 1, got {args.iso_tolerance}"
        )
    root = args.submission
    report = check_submission(root, args.iso_tolerance)
    summary = report["summary"]
    lines = [
        f"# Submission check: {report['status']}",
        "",
        f"{summary['passed_required_gates']} of {summary['required_gate_count']} required gates "
        f"passed, {_count(summary['warnings'], 'warning', 'warnings')}.",
        "",
        "| Result | Gate | Message |",
        "|---|---|---|",
        *(
            f"| {_result(gate)} | {gate['name']} | {gate['message'].replace('|', '/')} |"
            for gate in report["gates"]
        ),
    ]
    _write_atomic(root / "check_report.json", json.dumps(report, indent=2) + "\n")
    _write_atomic(root / "check_report.md", "\n".join(lines) + "\n")
    print(
        f"{report['status']}: {summary['passed_required_gates']} of "
        f"{summary['required_gate_count']} required gates passed, "
        f"{_count(summary['warnings'], 'warning', 'warnings')}; "
        f"report in {root / 'check_report.md'}",
        file=sys.stderr,
    )
    for gate in report["gates"]:
        if not gate["passed"]:
            print(f"{_result(gate)} {gate['name']}: {gate['message']}", file=sys.stderr)
    return 0 if report["status"] == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vesuvius.gp_submission",
        description="Cut column meshes for a submission and fail closed when the folder "
        "is incomplete.",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    def command(name: str, help_text: str) -> argparse.ArgumentParser:
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("--submission", required=True, type=Path, help="submission folder")
        return subparser

    subparser = command("cut-columns", "split a flattened tifxyz into one mesh per text column")
    subparser.add_argument("--surface", required=True, type=Path, help="flattened TIFXYZ directory")
    subparser.add_argument(
        "--cuts",
        required=True,
        type=Path,
        help="CSV with column,u0,u1[,kind]: grid columns of the surface, u1 exclusive",
    )
    subparser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the whole meshes/ folder, including files the tool did not write",
    )

    subparser = command("check", "write check_report.json/md; exit 2 if a required gate fails")
    subparser.add_argument(
        "--iso-tolerance",
        type=float,
        default=0.10,
        help="grid-edge stretch around 1.0 at p05 and p95 before mesh_isometry warns",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {"cut-columns": _cmd_cut_columns, "check": _cmd_check}
    try:
        return commands[args.cmd](args)
    except (SubmissionInputError, OSError, UnicodeDecodeError) as error:
        for line in str(error).splitlines():
            print(f"error: {line}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
