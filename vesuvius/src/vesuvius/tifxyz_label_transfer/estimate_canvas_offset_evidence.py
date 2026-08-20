#!/usr/bin/env python3
"""Estimate and cross-check a canvas offset from a prepared evidence manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from vesuvius.utils.cli import HyphenUnderscoreParser

from .core import load_affine
from .estimate_canvas_offset import estimate_canvas_offset
from .io import load_surface, read_image


def _tifxyz_resolution(path: Path) -> float:
    return float(path.name.split("um-", 1)[0])


def assess_evidence(
    reports: list[dict[str, Any]],
    *,
    minimum_evidence: int,
    maximum_disagreement_full_px: float,
) -> dict[str, Any]:
    accepted = [
        report
        for report in reports
        if report["converged"] and report["translation_model_valid"]
    ]
    offsets = np.asarray(
        [report["offset_yx_full_resolution_px"] for report in accepted],
        dtype=np.float64,
    )
    if offsets.size:
        consensus = np.median(offsets, axis=0)
        pairwise = offsets[:, None, :] - offsets[None, :, :]
        maximum_disagreement = float(
            np.max(np.linalg.norm(pairwise, axis=2))
        )
    else:
        consensus = np.asarray([math.nan, math.nan])
        maximum_disagreement = math.inf
    approved = (
        len(accepted) >= minimum_evidence
        and maximum_disagreement <= maximum_disagreement_full_px
    )
    return {
        "approved": approved,
        "accepted_evidence": len(accepted),
        "required_evidence": minimum_evidence,
        "canvas_offset_yx_full_resolution_px": consensus.tolist(),
        "maximum_evidence_disagreement_full_px": maximum_disagreement,
        "allowed_evidence_disagreement_full_px": (
            maximum_disagreement_full_px
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = HyphenUnderscoreParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-evidence", type=int, default=2)
    parser.add_argument(
        "--maximum-disagreement-full-px", type=float, default=3.0
    )
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--tolerance-px", type=float, default=0.25)
    parser.add_argument("--measure-tile-px", type=int, default=512)
    parser.add_argument("--min-coverage", type=float, default=0.6)
    parser.add_argument("--max-tiles", type=int, default=32)
    parser.add_argument("--nearest-vertices", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=65_536)
    parser.add_argument("--max-corner-drift-px", type=float, default=1.5)
    parser.add_argument(
        "--preprocess",
        choices=("dog", "gradient", "none"),
        default="dog",
    )
    parser.add_argument(
        "--stage-one-affine",
        "--same-volume-affine",
        dest="same_volume_affine",
    )
    parser.add_argument(
        "--stage-one-affine-direction",
        "--same-volume-affine-direction",
        dest="same_volume_affine_direction",
        choices=("forward", "inverse"),
        default="forward",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    case_dir = args.case_dir.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else case_dir / "renders" / "offset-evidence" / "manifest.json"
    )
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"error: {output} exists (pass --overwrite to replace)"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = load_surface(case_dir / "hf" / "source.tifxyz")
    targets = sorted(
        (case_dir / "open-data").glob("*um-*.tifxyz"),
        key=_tifxyz_resolution,
    )
    if not targets:
        raise ValueError(f"{case_dir} contains no target TIFXYZ")
    target_path = targets[0]
    target = load_surface(target_path)
    affine = None
    if args.same_volume_affine:
        affine = load_affine(args.same_volume_affine)
        if args.same_volume_affine_direction == "inverse":
            affine = np.linalg.inv(affine)

    reports: list[dict[str, Any]] = []
    for comparison in manifest["comparisons"]:
        source_render = read_image(comparison["source_render"])
        target_render = read_image(comparison["target_render"])
        print(f"Estimating evidence: {comparison['name']}")
        try:
            result = estimate_canvas_offset(
                source,
                target,
                source_render,
                target_render,
                max_iterations=args.max_iterations,
                tolerance_px=args.tolerance_px,
                measure_tile_px=args.measure_tile_px,
                min_coverage=args.min_coverage,
                max_tiles=args.max_tiles,
                nearest_vertices=args.nearest_vertices,
                tile_size=args.tile_size,
                query_batch_size=args.query_batch_size,
                affine=affine,
                preprocess=args.preprocess,
                max_corner_drift_px=args.max_corner_drift_px,
                verbose=True,
            )
        except ValueError as error:
            print(f"Evidence {comparison['name']} rejected: {error}")
            reports.append(
                {
                    "name": comparison["name"],
                    "source_render": comparison["source_render"],
                    "target_render": comparison["target_render"],
                    "source_render_shape": list(source_render.shape),
                    "target_render_shape": list(target_render.shape),
                    "converged": False,
                    "translation_model_valid": False,
                    "error": str(error),
                }
            )
            continue
        source_shape = np.asarray(source.full_resolution_shape)
        render_shape = np.asarray(source_render.shape)
        offset_full = (
            np.asarray(result["offset_yx_render_px"])
            * source_shape
            / render_shape
        )
        reports.append(
            {
                "name": comparison["name"],
                "source_render": comparison["source_render"],
                "target_render": comparison["target_render"],
                "source_render_shape": list(source_render.shape),
                "target_render_shape": list(target_render.shape),
                "offset_yx_full_resolution_px": offset_full.tolist(),
                **result,
            }
        )

    assessment = assess_evidence(
        reports,
        minimum_evidence=args.minimum_evidence,
        maximum_disagreement_full_px=(
            args.maximum_disagreement_full_px
        ),
    )
    document = {
        "tool": "estimate_canvas_offset_evidence.py",
        "manifest": str(manifest_path),
        "case_dir": str(case_dir),
        "source_tifxyz": str(case_dir / "hf" / "source.tifxyz"),
        "target_tifxyz": str(target_path),
        "same_volume_affine": args.same_volume_affine,
        "same_volume_affine_direction": (
            args.same_volume_affine_direction
            if args.same_volume_affine
            else None
        ),
        "preprocess": args.preprocess,
        **assessment,
        "evidence": reports,
        "convention": (
            "source render/label pixel (i, j) depicts source TIFXYZ canvas "
            "position (i + dy, j + dx); use the consensus full-resolution "
            "value only when approved is true"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(assessment, indent=2))
    print(f"Wrote {output}")
    return 0 if assessment["approved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
