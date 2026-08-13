"""Tests for tifxyz_topology_diagnostics.py — numpy/scipy only, no torch.

Tests the diagnostic script:
  - Validity mask: sentinel, NaN, z<=0 all detected
  - PCA thickness: flat plane -> no flag, folded -> flag
  - Holes: interior holes flagged, boundary holes ignored
  - Full pipeline integration
"""

import numpy as np
import pytest


# -- Validity mask tests -----------------------------------------------------

def test_validity_mask_sentinel():
    """Sentinel (-1, -1, -1) vertices should be marked invalid."""
    from tifxyz_topology_diagnostics import compute_validity_mask, DiagnosticConfig
    cfg = DiagnosticConfig()
    grid = np.ones((10, 10, 3), dtype=np.float32) * 5.0
    grid[3, 4] = [-1.0, -1.0, -1.0]  # sentinel
    valid = compute_validity_mask(grid, cfg)
    assert valid[3, 4] == False
    assert valid[0, 0] == True


def test_validity_mask_nan():
    """NaN in any coordinate should be marked invalid."""
    from tifxyz_topology_diagnostics import compute_validity_mask, DiagnosticConfig
    cfg = DiagnosticConfig()
    grid = np.ones((10, 10, 3), dtype=np.float32) * 5.0
    grid[2, 3, 1] = np.nan  # NaN in Y coordinate only
    valid = compute_validity_mask(grid, cfg)
    assert valid[2, 3] == False


def test_validity_mask_z_leq_zero():
    """z <= 0 should be marked invalid (Nieuwlaar fix, Jul 31 2026)."""
    from tifxyz_topology_diagnostics import compute_validity_mask, DiagnosticConfig
    cfg = DiagnosticConfig()
    grid = np.ones((10, 10, 3), dtype=np.float32) * 5.0
    grid[1, 1, 2] = 0.0   # z == 0 -> invalid
    grid[2, 2, 2] = -1.5  # z < 0 -> invalid
    grid[3, 3, 2] = 0.001 # z > 0 -> valid
    valid = compute_validity_mask(grid, cfg)
    assert valid[1, 1] == False, "z=0 should be invalid"
    assert valid[2, 2] == False, "z<0 should be invalid"
    assert valid[3, 3] == True, "z>0 should be valid"


def test_validity_mask_partial_sentinel_is_valid():
    """If only some coords are sentinel (e.g. x=-1, y=5, z=5), vertex is valid."""
    from tifxyz_topology_diagnostics import compute_validity_mask, DiagnosticConfig
    cfg = DiagnosticConfig()
    grid = np.ones((10, 10, 3), dtype=np.float32) * 5.0
    grid[4, 4, 0] = -1.0  # only x is sentinel, y and z are valid
    valid = compute_validity_mask(grid, cfg)
    # This should be valid because not ALL coords are sentinel
    assert valid[4, 4] == True


# -- PCA thickness tests -----------------------------------------------------

def test_pca_thickness_flat_plane_no_flag():
    """A perfectly flat plane should not trigger a sheet error."""
    from tifxyz_topology_diagnostics import detect_sheet_errors_pca_thickness
    H, W = 48, 48
    grid = np.zeros((H, W, 3), dtype=np.float32)
    # Flat plane at z=5
    for i in range(H):
        for j in range(W):
            grid[i, j] = [float(i), float(j), 5.0]
    valid = np.ones((H, W), dtype=bool)
    flags = detect_sheet_errors_pca_thickness(grid, valid, window_size=24,
                                              thickness_threshold=2.5, stride=8)
    assert len(flags) == 0, f"Flat plane should produce no flags, got {len(flags)}"


def test_pca_thickness_folded_surface_flags():
    """A surface with a fold (two z-levels) should trigger a sheet error."""
    from tifxyz_topology_diagnostics import detect_sheet_errors_pca_thickness
    H, W = 48, 48
    grid = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            # Top half at z=5, bottom half at z=15 (spanning two wraps)
            z = 5.0 if i < H // 2 else 15.0
            grid[i, j] = [float(i), float(j), z]
    valid = np.ones((H, W), dtype=bool)
    flags = detect_sheet_errors_pca_thickness(grid, valid, window_size=24,
                                              thickness_threshold=2.5, stride=8)
    # Should flag the transition region
    assert len(flags) > 0, "Folded surface should produce at least one flag"
    # All flags should be sheet_error_pca type
    assert all(f["type"] == "sheet_error_pca" for f in flags)


def test_pca_thickness_slightly_noisy_plane_no_flag():
    """A plane with small noise (< threshold) should not be flagged."""
    from tifxyz_topology_diagnostics import detect_sheet_errors_pca_thickness
    H, W = 48, 48
    rng = np.random.RandomState(42)
    grid = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            grid[i, j] = [float(i), float(j), 5.0 + rng.normal(0, 0.2)]
    valid = np.ones((H, W), dtype=bool)
    flags = detect_sheet_errors_pca_thickness(grid, valid, window_size=24,
                                              thickness_threshold=2.5, stride=8)
    assert len(flags) == 0, f"Noisy but flat plane should produce no flags, got {len(flags)}"


# -- Hole detection tests ----------------------------------------------------

def test_hole_detection_interior_hole():
    """An interior hole (invalid region not touching boundary) should be flagged."""
    from tifxyz_topology_diagnostics import detect_holes, DiagnosticConfig
    cfg = DiagnosticConfig(hole_min_area=2)
    valid = np.ones((20, 20), dtype=bool)
    # Create a 3x3 interior hole
    valid[8:11, 8:11] = False
    flags = detect_holes(valid, cfg)
    assert len(flags) == 1
    assert flags[0]["type"] == "hole"
    assert flags[0]["area_cells"] == 9


def test_hole_detection_boundary_not_flagged():
    """An invalid region touching the boundary should NOT be flagged as a hole."""
    from tifxyz_topology_diagnostics import detect_holes, DiagnosticConfig
    cfg = DiagnosticConfig(hole_min_area=2)
    valid = np.ones((20, 20), dtype=bool)
    # Create invalid region at boundary
    valid[0:3, 5:8] = False
    flags = detect_holes(valid, cfg)
    assert len(flags) == 0, "Boundary-touching invalid region should not be flagged"


def test_hole_detection_too_small_not_flagged():
    """Holes smaller than hole_min_area should not be flagged."""
    from tifxyz_topology_diagnostics import detect_holes, DiagnosticConfig
    cfg = DiagnosticConfig(hole_min_area=10)
    valid = np.ones((20, 20), dtype=bool)
    # Create a 2x2 interior hole (area = 4 < 10)
    valid[8:10, 8:10] = False
    flags = detect_holes(valid, cfg)
    assert len(flags) == 0, "Small hole should not be flagged"


# -- Integration test --------------------------------------------------------

def test_run_diagnostics_integration():
    """Full diagnostic pipeline should return expected structure."""
    from tifxyz_topology_diagnostics import run_diagnostics, DiagnosticConfig
    cfg = DiagnosticConfig()
    H, W = 64, 64
    grid = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            grid[i, j] = [float(i), float(j), 10.0]
    # Add one sentinel
    grid[5, 5] = [-1, -1, -1]

    result = run_diagnostics(grid, cfg)
    assert "summary" in result
    assert "correction_points" in result
    s = result["summary"]
    assert s["n_valid_vertices"] == H * W - 1
    assert s["n_missing_vertices"] == 1
    assert s["grid_shape"] == [H, W]
    assert 0.0 < s["valid_fraction"] <= 1.0


def test_run_diagnostics_output_is_json_serializable():
    """Output should be JSON-serializable for file writing."""
    import json
    from tifxyz_topology_diagnostics import run_diagnostics, DiagnosticConfig
    cfg = DiagnosticConfig()
    H, W = 32, 32
    grid = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            grid[i, j] = [float(i), float(j), 10.0]

    result = run_diagnostics(grid, cfg)
    # Should not raise
    serialized = json.dumps(result)
    assert len(serialized) > 0
