#!/usr/bin/env python3
"""Pre-screen tracks for winding-straddler paths using the psi field (C2b).

Straddler paths are surface tracks whose 3D points span more than one
physical winding of the papyrus scroll.  When fed into the spiral fitter
they pull the optimiser toward an averaged position between the two
windings, degrading fit quality.

This script uses the exported winding field psi(x) from C0 to identify
and flag straddler tracks before spiral fitting.  Tracks whose psi range
exceeds a threshold are marked for exclusion.

Expected straddler fraction: 5-20% in compressed regions.
If > 30%, the threshold is too low.

Usage:
    python psi_prescreening.py \\
        --tracks /path/to/tracks.dbm \\
        --psi /path/to/winding_field.zarr \\
        --output /path/to/prescreened_tracks.json \\
        --threshold 0.6

Input:
    - tracks.dbm: Track database (dbm format with pickled track data)
    - winding_field.zarr: Exported psi field from export_winding_field.py

Output:
    JSON file with track-level psi statistics and straddler classification.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import zarr  # type: ignore

log = logging.getLogger(__name__)


class PsiVolume:
    """Efficient psi field lookup from an OME-Zarr winding volume.

    Provides trilinear interpolation of the psi field at arbitrary 3D
    coordinates, mapping from full-resolution voxel coordinates to the
    (possibly downsampled) zarr grid.
    """

    def __init__(self, zarr_path: str):
        store = zarr.open(str(zarr_path), mode="r")
        if isinstance(store, zarr.Array):
            self.psi = np.asarray(store, dtype=np.float32)
        elif "winding_position" in store:
            self.psi = np.asarray(store["winding_position"], dtype=np.float32)
        else:
            self.psi = np.asarray(store, dtype=np.float32)
        self.resolution = int(store.attrs.get("resolution_factor",
                              store.attrs.get("scaledown", 4)))
        self.shape = self.psi.shape
        log.info(
            "PsiVolume loaded: shape=%s, resolution=%d, range=[%.2f, %.2f]",
            self.shape, self.resolution,
            float(np.nanmin(self.psi)), float(np.nanmax(self.psi)),
        )

    def query(self, zyx: np.ndarray) -> np.ndarray:
        """Look up psi values at full-resolution ZYX coordinates.

        Uses nearest-neighbour interpolation for speed. Points outside
        the volume boundary return NaN.

        Args:
            zyx: (N, 3) array of full-resolution voxel coordinates [z, y, x].

        Returns:
            (N,) array of psi values.
        """
        # Convert from full-res to zarr-grid coordinates
        grid_coords = (zyx / self.resolution).astype(np.int32)

        # Clip to valid range, mark out-of-bounds
        result = np.full(len(zyx), np.nan, dtype=np.float32)
        valid = np.ones(len(zyx), dtype=bool)
        for dim in range(3):
            valid &= (grid_coords[:, dim] >= 0) & (grid_coords[:, dim] < self.shape[dim])

        if valid.any():
            coords = grid_coords[valid]
            result[valid] = self.psi[coords[:, 0], coords[:, 1], coords[:, 2]]

        return result


def analyze_track_psi(
    track_zyx: np.ndarray,
    psi_vol: PsiVolume,
) -> dict:
    """Compute psi statistics for a single track.

    Args:
        track_zyx: (N, 3) array of track point coordinates.
        psi_vol: PsiVolume instance.

    Returns:
        Dict with psi statistics for the track.
    """
    psi_values = psi_vol.query(track_zyx)
    valid_psi = psi_values[np.isfinite(psi_values)]

    if len(valid_psi) == 0:
        return {
            "n_points": len(track_zyx),
            "n_valid_psi": 0,
            "psi_min": None,
            "psi_max": None,
            "psi_range": None,
            "psi_mean": None,
            "psi_std": None,
            "is_straddler": False,
            "note": "No valid psi values (track outside winding volume)",
        }

    psi_min = float(valid_psi.min())
    psi_max = float(valid_psi.max())
    psi_range = psi_max - psi_min
    psi_mean = float(valid_psi.mean())
    psi_std = float(valid_psi.std())

    return {
        "n_points": len(track_zyx),
        "n_valid_psi": int(len(valid_psi)),
        "psi_min": round(psi_min, 4),
        "psi_max": round(psi_max, 4),
        "psi_range": round(psi_range, 4),
        "psi_mean": round(psi_mean, 4),
        "psi_std": round(psi_std, 4),
    }


def prescreen_tracks(
    tracks: dict[str, np.ndarray],
    psi_vol: PsiVolume,
    threshold: float = 0.6,
) -> dict:
    """Classify tracks as clean or straddler based on psi range.

    Args:
        tracks: Dict mapping track_id -> (N, 3) ZYX array.
        psi_vol: PsiVolume instance.
        threshold: Maximum allowed psi range for a clean track.
            Tracks with psi_range > threshold are classified as straddlers.
            Default 0.6 = just over half a winding.

    Returns:
        Dict with per-track results and summary statistics.
    """
    results = {}
    n_straddler = 0
    n_clean = 0
    n_unknown = 0

    for track_id, track_zyx in tracks.items():
        stats = analyze_track_psi(track_zyx, psi_vol)

        if stats["psi_range"] is None:
            stats["is_straddler"] = False
            stats["classification"] = "unknown"
            n_unknown += 1
        elif stats["psi_range"] > threshold:
            stats["is_straddler"] = True
            stats["classification"] = "straddler"
            n_straddler += 1
        else:
            stats["is_straddler"] = False
            stats["classification"] = "clean"
            n_clean += 1

        results[track_id] = stats

    total = n_clean + n_straddler + n_unknown
    straddler_frac = n_straddler / max(total - n_unknown, 1)

    summary: dict[str, Any] = {
        "total_tracks": total,
        "clean_tracks": n_clean,
        "straddler_tracks": n_straddler,
        "unknown_tracks": n_unknown,
        "straddler_fraction": round(straddler_frac, 4),
        "threshold": threshold,
    }

    # Sanity warnings
    if straddler_frac > 0.30:
        summary["warning"] = (
            f"Straddler fraction {straddler_frac:.1%} > 30%. "
            f"Threshold {threshold} may be too low."
        )
        log.warning(summary["warning"])
    elif straddler_frac < 0.01 and total > 100:
        summary["warning"] = (
            f"Straddler fraction {straddler_frac:.1%} < 1%. "
            f"Threshold {threshold} may be too high, or psi field may not cover these tracks."
        )
        log.warning(summary["warning"])

    log.info(
        "Pre-screening: %d tracks | %d clean (%.1f%%) | %d straddlers (%.1f%%) | %d unknown",
        total, n_clean, 100 * n_clean / max(total, 1),
        n_straddler, 100 * straddler_frac,
        n_unknown,
    )

    return {"summary": summary, "tracks": results}


def load_tracks_from_dbm(dbm_path: str) -> dict[str, np.ndarray]:
    """Load tracks from a .dbm track database.

    Each track is stored as a pickled array of ZYX coordinates.

    Args:
        dbm_path: Path to the tracks .dbm file.

    Returns:
        Dict mapping track_id -> (N, 3) numpy array.
    """
    import dbm
    import pickle

    tracks = {}
    db = dbm.open(str(dbm_path), "r")
    for key in db.keys():
        try:
            data = pickle.loads(db[key])
            if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 3:
                tracks[key.decode() if isinstance(key, bytes) else key] = data[:, :3]
            elif isinstance(data, dict) and "points_zyx" in data:
                tracks[key.decode() if isinstance(key, bytes) else key] = np.array(data["points_zyx"])
        except Exception as e:
            log.debug("Skipping track %s: %s", key, e)
    db.close()

    log.info("Loaded %d tracks from %s", len(tracks), dbm_path)
    return tracks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(
        description="Pre-screen tracks for winding straddlers using the psi field."
    )
    p.add_argument("--tracks", required=True, help="Path to tracks.dbm")
    p.add_argument("--psi", required=True, help="Path to winding_field.zarr")
    p.add_argument("--output", "-o", required=True, help="Output JSON path")
    p.add_argument("--threshold", type=float, default=0.6,
                   help="Max psi range for clean tracks (default: 0.6)")
    args = p.parse_args()

    psi_vol = PsiVolume(args.psi)
    tracks = load_tracks_from_dbm(args.tracks)
    result = prescreen_tracks(tracks, psi_vol, threshold=args.threshold)

    Path(args.output).write_text(json.dumps(result, indent=2))
    log.info("Results written to %s", args.output)
