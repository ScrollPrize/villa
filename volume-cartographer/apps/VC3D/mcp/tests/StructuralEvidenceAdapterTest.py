#!/usr/bin/env python3
"""Synthetic regression tests for Phase 3 structural evidence."""
from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import zarr


def load_adapter(path: Path):
    spec = importlib.util.spec_from_file_location("structural_evidence_adapter", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registered(root: Path, signal: np.ndarray) -> Path:
    artifact = root
    group = zarr.open_group(str(artifact / "surface.zarr"), mode="w", zarr_format=2)
    geometry = group.require_group("geometry"); renders = group.require_group("renders")
    v, u = np.mgrid[: signal.shape[0], : signal.shape[1]]
    xyz = np.stack((u, v, np.full_like(u, 30)), axis=-1).astype(np.float32)
    chunks = (64, 64)
    geometry.create_array("xyz", data=xyz, chunks=(*chunks, 3))
    geometry.create_array("valid", data=np.ones(signal.shape, np.uint8), chunks=chunks)
    renders.create_array("raw", data=signal.astype(np.float32), chunks=chunks)
    group.attrs["registered_volume"] = {
        "voxel_spacing": [100.0, 100.0, 100.0],
        "voxel_spacing_unit": "um",
        "voxel_spacing_explicit": True,
    }
    (artifact / "manifest.json").write_text(json.dumps({"kind": "vc_registered_surface_v1"}))
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--adapter", required=True, type=Path); args = parser.parse_args()
    adapter = load_adapter(args.adapter)
    with tempfile.TemporaryDirectory(prefix="vc-structural-evidence-") as temporary:
        root = Path(temporary); v, u = np.mgrid[:256, :256]
        base = 0.5 + 0.22 * np.sin(2 * np.pi * u / 30.0) + 0.18 * np.sin(2 * np.pi * v / 40.0)
        a = registered(root / "a", base.astype(np.float32))
        altered = base.copy(); altered[:, 128:] += 0.12 * np.sin(2 * np.pi * u[:, 128:] / 17.0)
        b = registered(root / "b", altered.astype(np.float32))
        common = {
            "polarity": "bright", "letter_period_mm": 3.0, "line_period_mm": 4.0,
            "window_width_mm": 15.0, "window_height_mm": 12.0, "step_mm": 4.0,
            "minimum_cycles": 3.0, "null_trials": 4, "null_seed": 7,
        }
        grid_request = {**common, "surface": {"job_id": "a", "artifact_id": "registered-surface"}}
        grid_out = root / "grid"; grid = adapter.grid_coherence(grid_request, a, grid_out)
        assert grid["periods_mm"]["letters"] == 3.0
        assert grid["periods_mm"]["lines"] == 4.0
        assert grid["score_summary"]["maximum"] > 0
        assert 0 < grid["significance"]["empirical_p_value"] <= 1
        assert (grid_out / "grid-coherence.zarr" / ".zgroup").is_file()

        compare_request = {
            **common,
            "surface_a": {"job_id": "a", "artifact_id": "registered-surface"},
            "surface_b": {"job_id": "b", "artifact_id": "registered-surface"},
        }
        comparison_out = root / "comparison"; comparison = adapter.compare_registered(compare_request, a, b, comparison_out)
        assert comparison["maximum_xyz_difference"] == 0.0
        assert comparison["map_correlation"] <= 1.0
        assert (comparison_out / "structural-divergence.png").is_file()

        fold_request = {
            "surface": {"job_id": "a", "artifact_id": "registered-surface"},
            "grid": {"job_id": "grid", "artifact_id": "grid-coherence"},
            "polarity": "bright", "period_tolerance": 0.1, "period_steps": 21,
            "phase_bins": 32, "null_trials": 4, "null_seed": 11,
        }
        fold_out = root / "fold"; folded = adapter.epoch_fold(fold_request, a, grid_out, fold_out)
        assert abs(folded["best_period_mm"] - 4.0) < 0.5
        assert 0 < folded["look_elsewhere_corrected_empirical_p_value"] <= 1
        assert (fold_out / "folded-phase-profile.png").is_file()
    print("StructuralEvidenceAdapterTest passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
