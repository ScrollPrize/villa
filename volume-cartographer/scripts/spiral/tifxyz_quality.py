#!/usr/bin/env python3
"""Classify graph-grown TIFXYZ surfaces without confusing holes with folds."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import functools
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable


# Conservative defaults derived from the graph_patches population.  The
# thickness test is the primary mixed-wrap signal.  Cleanup damage is a
# secondary signal and intentionally requires both extreme trimming and folds,
# so ordinary interior holes do not count against a surface.
DEFAULT_MAX_THICK_CELL_FRAC = 0.05
DEFAULT_MAX_TRIMMED_FRAC = 0.25
DEFAULT_MIN_FOLD_MASKED_FRAC = 0.02


@dataclasses.dataclass(frozen=True)
class TifxyzAssessment:
    path: Path
    status: str
    thick_cell_frac: float | None
    trimmed_fraction: float | None
    fold_masked_fraction: float | None
    reasons: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "status": self.status,
            "thick_cell_frac": self.thick_cell_frac,
            "trimmed_fraction": self.trimmed_fraction,
            "fold_masked_fraction": self.fold_masked_fraction,
            "reasons": list(self.reasons),
        }


def _finite_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def assess_metadata(
    path: Path,
    metadata: dict,
    *,
    max_thick_cell_frac: float = DEFAULT_MAX_THICK_CELL_FRAC,
    max_trimmed_frac: float = DEFAULT_MAX_TRIMMED_FRAC,
    min_fold_masked_frac: float = DEFAULT_MIN_FOLD_MASKED_FRAC,
    reject_any_fold_fixes: bool = False,
) -> TifxyzAssessment:
    """Assess generator diagnostics; surface-mask holes are deliberately unused."""
    quality = metadata.get("quality")
    raster = metadata.get("raster")
    quality = quality if isinstance(quality, dict) else {}
    raster = raster if isinstance(raster, dict) else {}
    slim = raster.get("slim")
    slim = slim if isinstance(slim, dict) else {}

    thick = _finite_number(quality.get("thick_cell_frac"))
    trimmed = _finite_number(raster.get("trimmed_fraction"))
    fold_masked = _finite_number(raster.get("fold_masked_vertices"))
    slim_fold_masked = _finite_number(slim.get("fold_masked_vertices"))
    valid = _finite_number(raster.get("valid_vertices"))
    fold_fraction = None
    if fold_masked is not None and valid is not None and fold_masked + valid > 0:
        fold_fraction = fold_masked / (fold_masked + valid)

    reasons = []
    if thick is not None and thick >= max_thick_cell_frac:
        reasons.append(
            f"mixed-sheet cells {thick:.2%} >= {max_thick_cell_frac:.2%}"
        )
    if reject_any_fold_fixes and (
        (fold_masked is not None and fold_masked > 0)
        or (slim_fold_masked is not None and slim_fold_masked > 0)
    ):
        reasons.append(
            "fold fixes applied "
            f"(raster={int(fold_masked or 0)}, slim={int(slim_fold_masked or 0)})"
        )
    if (
        trimmed is not None
        and fold_fraction is not None
        and trimmed >= max_trimmed_frac
        and fold_fraction >= min_fold_masked_frac
    ):
        reasons.append(
            f"cleanup damage: trimmed {trimmed:.2%} and fold-masked "
            f"{fold_fraction:.2%}"
        )

    if reasons:
        status = "reject"
    elif thick is None:
        # Old/arbitrary TIFXYZs cannot be certified from their final validity
        # mask: a genuine hole and a hole cut to remove a fold look identical.
        status = "unknown"
        reasons.append("missing quality.thick_cell_frac")
    else:
        status = "accept"

    return TifxyzAssessment(
        path=path,
        status=status,
        thick_cell_frac=thick,
        trimmed_fraction=trimmed,
        fold_masked_fraction=fold_fraction,
        reasons=tuple(reasons),
    )


def assess_tifxyz(
    path: str | Path,
    *,
    max_thick_cell_frac: float = DEFAULT_MAX_THICK_CELL_FRAC,
    max_trimmed_frac: float = DEFAULT_MAX_TRIMMED_FRAC,
    min_fold_masked_frac: float = DEFAULT_MIN_FOLD_MASKED_FRAC,
    reject_any_fold_fixes: bool = False,
) -> TifxyzAssessment:
    path = Path(path)
    try:
        with (path / "meta.json").open() as stream:
            metadata = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        return TifxyzAssessment(
            path, "unknown", None, None, None, (f"cannot read meta.json: {error}",)
        )
    if not isinstance(metadata, dict):
        return TifxyzAssessment(
            path, "unknown", None, None, None, ("meta.json is not an object",)
        )
    return assess_metadata(
        path,
        metadata,
        max_thick_cell_frac=max_thick_cell_frac,
        max_trimmed_frac=max_trimmed_frac,
        min_fold_masked_frac=min_fold_masked_frac,
        reject_any_fold_fixes=reject_any_fold_fixes,
    )


def iter_tifxyz(inputs: Iterable[str | Path]) -> Iterable[Path]:
    """Yield explicit TIFXYZs and TIFXYZ children of input directories once."""
    seen = set()
    for raw in inputs:
        path = Path(raw)
        if path.name.endswith(".tifxyz"):
            candidates = (path,)
        elif path.is_dir():
            candidates = path.rglob("*.tifxyz")
        else:
            candidates = ()
        for candidate in candidates:
            key = str(candidate.resolve())
            if key not in seen:
                seen.add(key)
                yield candidate


def _fraction(text: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number in [0, 1]") from error
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return value


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Find mixed/folded graph-grown TIFXYZ surfaces using provenance "
            "diagnostics. Interior holes alone are not considered defects."
        )
    )
    parser.add_argument("paths", nargs="+", help="TIFXYZ or directory to scan")
    parser.add_argument(
        "--max-thick-cell-frac",
        type=_fraction,
        default=DEFAULT_MAX_THICK_CELL_FRAC,
        help="reject at this mixed-sheet cell fraction (default: %(default)s)",
    )
    parser.add_argument(
        "--max-trimmed-frac",
        type=_fraction,
        default=DEFAULT_MAX_TRIMMED_FRAC,
        help="secondary cleanup-damage threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--min-fold-masked-frac",
        type=_fraction,
        default=DEFAULT_MIN_FOLD_MASKED_FRAC,
        help="fold fraction paired with --max-trimmed-frac (default: %(default)s)",
    )
    parser.add_argument(
        "--reject-any-fold-fixes",
        action="store_true",
        help=(
            "independently reject a patch if either rasterization stage "
            "masked one or more folded vertices"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="print accepted surfaces too (rejects and unknowns print by default)",
    )
    parser.add_argument("--jsonl", action="store_true", help="emit JSON Lines")
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=min(32, (os.cpu_count() or 1) * 2),
        help="concurrent metadata readers (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    counts = {"accept": 0, "reject": 0, "unknown": 0}
    paths = list(iter_tifxyz(args.paths))
    if not paths:
        parser.error("no .tifxyz directories found")
    assess = functools.partial(
        assess_tifxyz,
        max_thick_cell_frac=args.max_thick_cell_frac,
        max_trimmed_frac=args.max_trimmed_frac,
        min_fold_masked_frac=args.min_fold_masked_frac,
        reject_any_fold_fixes=args.reject_any_fold_fixes,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        assessments = executor.map(
            assess,
            paths,
            buffersize=max(64, args.workers * 4),
        )
        for assessment in assessments:
            counts[assessment.status] += 1
            if not args.all and assessment.status == "accept":
                continue
            if args.jsonl:
                print(json.dumps(assessment.as_dict(), sort_keys=True))
            else:
                detail = "; ".join(assessment.reasons) or "quality checks passed"
                print(f"{assessment.status:7} {assessment.path}  {detail}")

    print(
        f"scanned={sum(counts.values())} reject={counts['reject']} "
        f"unknown={counts['unknown']} accept={counts['accept']}",
        file=sys.stderr,
    )
    return 1 if counts["reject"] or counts["unknown"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
