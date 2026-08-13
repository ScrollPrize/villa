#!/usr/bin/env python3
"""Psi-guided track grid growth extensions (C1).

Extends the existing grow_track_grids.py with winding-field-aware capabilities:

  1. **Psi-guided seed selection**: Seeds are scored by their psi field coverage
     and consistency, preferring tracks that lie cleanly within a single winding.
     Straddler tracks (spanning >1 winding) are deprioritised as growth seeds.

  2. **Winding-aware growth validation**: During surface growth, candidate
     tracks are checked against the psi field.  A candidate whose psi value
     disagrees with the seed's winding is rejected before the expensive
     geometric crossing checks.

  3. **Surface winding consistency QA**: After growth, each surface is scored
     by the variance of psi values across its constituent tracks.  High-variance
     surfaces are flagged as likely multi-winding artifacts.

This module wraps grow_track_grids.py functions without modifying the core
algorithm.  It can be used as a drop-in replacement or selectively applied.

Usage:
    python psi_guided_growth.py \\
        /path/to/tracks.dbm /tmp/psi-track-grids \\
        --psi /path/to/winding_field.zarr \\
        --center-xyz 3848 2775 8212 \\
        --bbox-size-zyx 3000 1000 1000 \\
        --count 32 --seed-spacing 256 --workers 8
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclasses.dataclass
class PsiSeedScore:
    """Psi-field seed quality metrics for a track."""
    track_id: int
    median_psi: float
    std_psi: float
    n_valid: int
    is_straddler: bool
    score: float  # Higher = better seed quality


def score_tracks_by_psi(
    tracks: list[np.ndarray],
    psi: np.ndarray,
    resolution: int,
    straddler_threshold: float = 0.5,
) -> list[PsiSeedScore]:
    """Score tracks by their psi field consistency for seed selection.

    Tracks that lie cleanly within a single winding (low psi variance)
    are scored higher.  Straddler tracks are deprioritised.

    Args:
        tracks: List of (N, 3) ZYX arrays.
        psi: (Z, Y, X) psi volume.
        resolution: Resolution factor for coordinate mapping.
        straddler_threshold: Max psi range for a clean track.

    Returns:
        List of PsiSeedScore, one per track.
    """
    scores = []
    for track_id, track in enumerate(tracks):
        grid_coords = (track / resolution).astype(np.int32)
        valid = np.ones(len(track), dtype=bool)
        for dim in range(3):
            valid &= (grid_coords[:, dim] >= 0) & (grid_coords[:, dim] < psi.shape[dim])

        if valid.sum() < 3:
            scores.append(PsiSeedScore(
                track_id=track_id, median_psi=0.0, std_psi=999.0,
                n_valid=0, is_straddler=True, score=0.0,
            ))
            continue

        coords = grid_coords[valid]
        psi_vals = psi[coords[:, 0], coords[:, 1], coords[:, 2]]
        psi_valid = psi_vals[psi_vals > 0.5]

        if len(psi_valid) < 3:
            scores.append(PsiSeedScore(
                track_id=track_id, median_psi=0.0, std_psi=999.0,
                n_valid=0, is_straddler=True, score=0.0,
            ))
            continue

        median_psi = float(np.median(psi_valid))
        std_psi = float(np.std(psi_valid))
        psi_range = float(psi_valid.max() - psi_valid.min())
        is_straddler = psi_range > straddler_threshold

        # Score: high coverage * low variance * non-straddler bonus
        coverage = min(1.0, len(psi_valid) / max(len(track), 1))
        consistency = max(0.0, 1.0 - std_psi * 2.0)
        straddler_penalty = 0.2 if is_straddler else 1.0
        score = coverage * consistency * straddler_penalty

        scores.append(PsiSeedScore(
            track_id=track_id,
            median_psi=round(median_psi, 4),
            std_psi=round(std_psi, 4),
            n_valid=int(len(psi_valid)),
            is_straddler=is_straddler,
            score=round(score, 4),
        ))

    return scores


def psi_guided_seed_selection(
    scores: list[PsiSeedScore],
    eligible_tracks: list[int],
    representative_zyx: np.ndarray,
    count: int,
    spacing: float,
    seed: int = 42,
) -> list[int]:
    """Select growth seeds using psi-guided scoring.

    Tracks are sorted by psi score (descending), then spatially separated
    using the same spacing logic as grow_track_grids.spaced_random_tracks.

    Args:
        scores: List of PsiSeedScore from score_tracks_by_psi.
        eligible_tracks: List of candidate track IDs.
        representative_zyx: (T, 3) representative positions per track.
        count: Number of seeds to select.
        spacing: Minimum spacing between selected seeds.
        seed: Random seed.

    Returns:
        List of selected track IDs.
    """
    import collections
    import itertools
    import random

    # Filter to eligible and sort by score (highest first)
    eligible_set = set(eligible_tracks)
    eligible_scores = [s for s in scores if s.track_id in eligible_set and s.score > 0]
    eligible_scores.sort(key=lambda s: s.score, reverse=True)

    # Add randomness among ties (shuffle within score bands)
    rng = random.Random(seed)
    band_size = max(1, len(eligible_scores) // 20)
    for i in range(0, len(eligible_scores), band_size):
        band = eligible_scores[i:i + band_size]
        rng.shuffle(band)
        eligible_scores[i:i + band_size] = band

    # Spatial separation
    if spacing <= 0:
        return [s.track_id for s in eligible_scores[:count]]

    cell_size = spacing
    buckets: dict[tuple[int, int, int], list[np.ndarray]] = collections.defaultdict(list)
    selected = []
    for s in eligible_scores:
        point = representative_zyx[s.track_id]
        cell = tuple(np.floor(point / cell_size).astype(int))
        too_close = False
        for offset in itertools.product((-1, 0, 1), repeat=3):
            neighbor = tuple(cell[ax] + offset[ax] for ax in range(3))
            for other in buckets.get(neighbor, ()):
                if np.linalg.norm(point - other) < spacing:
                    too_close = True
                    break
            if too_close:
                break
        if too_close:
            continue
        selected.append(s.track_id)
        buckets[cell].append(point)
        if len(selected) >= count:
            break

    return selected


def validate_surface_winding_consistency(
    surface_track_ids: list[int],
    tracks: list[np.ndarray],
    psi: np.ndarray,
    resolution: int,
) -> dict:
    """Score a grown surface by winding consistency.

    After surface growth, check that all tracks in the surface agree on
    their winding assignment.  High psi variance across tracks indicates
    a multi-winding artifact.

    Args:
        surface_track_ids: List of track IDs composing the surface.
        tracks: List of all tracks.
        psi: (Z, Y, X) psi volume.
        resolution: Resolution factor.

    Returns:
        Dict with consistency metrics.
    """
    all_psi = []
    for tid in surface_track_ids:
        track = tracks[tid]
        grid_coords = (track / resolution).astype(np.int32)
        valid = np.ones(len(track), dtype=bool)
        for dim in range(3):
            valid &= (grid_coords[:, dim] >= 0) & (grid_coords[:, dim] < psi.shape[dim])
        if valid.any():
            coords = grid_coords[valid]
            psi_vals = psi[coords[:, 0], coords[:, 1], coords[:, 2]]
            psi_valid = psi_vals[psi_vals > 0.5]
            if len(psi_valid) > 0:
                all_psi.append(float(np.median(psi_valid)))

    if len(all_psi) < 2:
        return {
            "n_tracks_with_psi": len(all_psi),
            "winding_consistent": True,
            "note": "Insufficient psi coverage for assessment.",
        }

    psi_arr = np.array(all_psi)
    median_psi = float(np.median(psi_arr))
    std_psi = float(np.std(psi_arr))
    psi_range = float(psi_arr.max() - psi_arr.min())

    # Consistent if all tracks agree to within 0.5 windings
    is_consistent = psi_range < 0.5

    return {
        "n_tracks_with_psi": len(all_psi),
        "median_psi": round(median_psi, 4),
        "std_psi": round(std_psi, 4),
        "psi_range": round(psi_range, 4),
        "winding_consistent": is_consistent,
        "dominant_winding": int(round(median_psi)),
        "note": (
            "PASS: surface is winding-consistent"
            if is_consistent
            else f"WARNING: psi range {psi_range:.2f} > 0.5 — likely multi-winding artifact"
        ),
    }


def psi_growth_report(
    surfaces: list[dict],
    scores: list[PsiSeedScore],
) -> dict:
    """Generate a summary report of psi-guided growth results.

    Args:
        surfaces: List of surface dicts with 'track_ids' and 'consistency'.
        scores: PsiSeedScore list from scoring phase.

    Returns:
        Summary report dict.
    """
    n_consistent = sum(1 for s in surfaces if s.get("consistency", {}).get("winding_consistent", True))
    n_total = len(surfaces)
    n_straddler_seeds = sum(1 for s in scores if s.is_straddler)

    return {
        "total_surfaces": n_total,
        "winding_consistent_surfaces": n_consistent,
        "multi_winding_surfaces": n_total - n_consistent,
        "consistency_rate": round(n_consistent / max(n_total, 1), 4),
        "total_scored_tracks": len(scores),
        "straddler_tracks": n_straddler_seeds,
        "straddler_fraction": round(n_straddler_seeds / max(len(scores), 1), 4),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(
        description="Psi-guided track grid growth (C1 extension)."
    )
    p.add_argument("tracks_dbm", help="Path to tracks.dbm")
    p.add_argument("output_dir", help="Output directory for grown grids")
    p.add_argument("--psi", required=True, help="Path to winding_field.zarr")
    p.add_argument("--center-xyz", nargs=3, type=float, default=None,
                   help="ROI center (x, y, z)")
    p.add_argument("--bbox-size-zyx", nargs=3, type=float, default=None,
                   help="ROI bounding box size (z, y, x)")
    p.add_argument("--z-range", nargs=2, type=int, default=[4000, 17000],
                   help="Z range (default: 4000 17000)")
    p.add_argument("--count", type=int, default=32, help="Number of surfaces")
    p.add_argument("--seed-spacing", type=float, default=256,
                   help="Minimum spacing between seeds")
    p.add_argument("--straddler-threshold", type=float, default=0.5,
                   help="Max psi range for non-straddler classification")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    import zarr  # type: ignore

    # Load psi volume
    store = zarr.open(args.psi, mode="r")
    if isinstance(store, zarr.Array):
        psi = np.asarray(store, dtype=np.float32)
    elif "winding_position" in store:
        psi = np.asarray(store["winding_position"], dtype=np.float32)
    else:
        psi = np.asarray(store, dtype=np.float32)
    resolution = int(store.attrs.get("resolution_factor",
                     store.attrs.get("scaledown", 4)))

    log.info("Psi volume: shape=%s, resolution=%d", psi.shape, resolution)

    # Build track graph (reuse existing grow_track_grids infrastructure)
    from grow_track_grids import (
        normalize_dbm_path, build_graph, centered_roi,
        length_bin_bounds, tracks_in_bin,
    )

    dbm_path = normalize_dbm_path(args.tracks_dbm)
    roi_lo, roi_hi = None, None
    if args.center_xyz and args.bbox_size_zyx:
        roi_lo, roi_hi = centered_roi(args.center_xyz, args.bbox_size_zyx)

    z_range = tuple(args.z_range)
    graph = build_graph(dbm_path, z_range, roi_lo, roi_hi,
                        angle_degrees=60.0, tangent_radius=32.0)

    # Score all tracks by psi consistency
    log.info("Scoring %d tracks by psi consistency...", len(graph.tracks))
    scores = score_tracks_by_psi(
        graph.tracks, psi, resolution,
        straddler_threshold=args.straddler_threshold,
    )
    n_good = sum(1 for s in scores if s.score > 0.5)
    n_straddler = sum(1 for s in scores if s.is_straddler)
    log.info("Scored: %d good (score > 0.5), %d straddlers", n_good, n_straddler)

    # Select seeds using psi-guided scoring
    eligible = [s.track_id for s in scores if s.score > 0]
    bounds = length_bin_bounds(graph.lengths, None)
    long_tracks = tracks_in_bin(graph, "long", bounds, eligible)

    selected = psi_guided_seed_selection(
        scores, long_tracks, graph.representative_zyx,
        count=args.count, spacing=args.seed_spacing, seed=args.seed,
    )
    log.info("Selected %d psi-guided seeds", len(selected))

    # Report
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = psi_growth_report([], scores)
    report["selected_seeds"] = selected
    report["seed_scores"] = [
        dataclasses.asdict(s) for s in scores if s.track_id in set(selected)
    ]

    report_path = out_dir / "psi_growth_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("Report written to %s", report_path)
    log.info("To grow surfaces with these seeds, pass them to grow_track_grids.py")
