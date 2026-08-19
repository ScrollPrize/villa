"""Tests for the metadata self-consistency check in repair_tifxyz_spacing.py.

Anchored on the real numbers from the issue this addresses: PHercParis4's
published `outer_shell` declares scale ~[19.997318, 19.996687] against a
measured ~20-voxel grid -- a ~400x error that the existing --target-spacing
check misses whenever the geometry happens to measure close to the default
target (20.0), because that check never looks at the file's own `scale`
field at all.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tifffile

from repair_tifxyz_spacing import (
    fix_metadata_scale,
    load_scale,
    measure_spacing,
    tifxyz_dirs,
)


def make_tifxyz(root, name, scale, step, grid=(60, 68)):
    """A synthetic tifxyz with a real `step`-voxel grid and an independently
    declared `scale` -- exactly like a real file where the two can disagree.
    """
    h, w = grid
    d = root / name
    d.mkdir(parents=True)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) * step
    z = np.full((h, w), 500.0, np.float32)
    for arr, nm in ((xx, "x"), (yy, "y"), (z, "z")):
        tifffile.imwrite(d / f"{nm}.tif", arr)
    meta = {"uuid": name, "scale": list(scale), "format": "tifxyz", "type": "seg"}
    (d / "meta.json").write_text(json.dumps(meta))
    return d


# --- the headline case: reproduces the real published bug -----------------

def test_reproduces_the_reported_outer_shell_case(tmp_path):
    """Declared scale ~19.997 against a real ~20-voxel grid: a ~400x error."""
    d = make_tifxyz(tmp_path, "outer_shell", scale=(19.997318, 19.996687), step=20.49)
    s = measure_spacing(d, tmp_path, target_spacing=20.0, threshold=0.15)

    assert s.needs_repair is False, (
        "this is the actual bug: geometry measures close to the DEFAULT "
        "target-spacing, so the old check alone reports the file as fine"
    )
    assert s.metadata_mismatch is True
    assert s.metadata_factor == pytest.approx(409.7, rel=0.02)  # issue reports ~400x


def test_conforming_sibling_in_the_same_pack_is_clean(tmp_path):
    """A correctly-declared file (scale = 1/step) must not be flagged."""
    d = make_tifxyz(tmp_path, "verified_patch", scale=(0.05, 0.05), step=20.0)
    s = measure_spacing(d, tmp_path, target_spacing=20.0, threshold=0.15)
    assert s.needs_repair is False
    assert s.metadata_mismatch is False
    assert s.metadata_factor == pytest.approx(1.0, abs=0.05)


# --- the check is independent of --target-spacing --------------------------

def test_metadata_check_does_not_depend_on_target_spacing(tmp_path):
    """The whole point: this check needs no external guess about spacing."""
    d = make_tifxyz(tmp_path, "outer_shell", scale=(19.997318, 19.996687), step=20.49)
    for guess in (5.0, 20.0, 50.0, 100.0):
        s = measure_spacing(d, tmp_path, target_spacing=guess, threshold=0.15)
        assert s.metadata_mismatch is True, f"should still catch it at target_spacing={guess}"
        assert s.metadata_factor == pytest.approx(409.7, rel=0.02), (
            "metadata_factor must not move with target_spacing -- it compares "
            "the file against ITSELF, not against the guess"
        )


def test_metadata_threshold_is_independently_configurable(tmp_path):
    """A mild disagreement (10%) should not trip the default 15% threshold,
    but should trip a tighter one -- independent of the repair threshold."""
    d = make_tifxyz(tmp_path, "mild", scale=(1 / 22.0, 1 / 22.0), step=20.0)
    loose = measure_spacing(d, tmp_path, target_spacing=20.0, threshold=0.15,
                            metadata_threshold=0.15)
    tight = measure_spacing(d, tmp_path, target_spacing=20.0, threshold=0.15,
                            metadata_threshold=0.05)
    assert loose.metadata_mismatch is False
    assert tight.metadata_mismatch is True


# --- fix_metadata_scale: geometry must be untouched -------------------------

def test_fix_metadata_scale_writes_only_scale(tmp_path):
    d = make_tifxyz(tmp_path, "outer_shell", scale=(19.997318, 19.996687), step=20.49)
    x_before = tifffile.imread(d / "x.tif").copy()

    s = measure_spacing(d, tmp_path, target_spacing=20.0, threshold=0.15)
    new_sx, new_sy = fix_metadata_scale(d / "meta.json", s.median_right, s.median_down)

    x_after = tifffile.imread(d / "x.tif")
    assert np.array_equal(x_before, x_after), "geometry must not be touched"

    assert new_sx == pytest.approx(1.0 / 20.49, rel=1e-6)
    sx2, sy2 = load_scale(d / "meta.json")
    assert (sx2, sy2) == (new_sx, new_sy)


def test_fix_metadata_scale_result_is_self_consistent(tmp_path):
    """After the fix, re-measuring the same file must report no mismatch."""
    d = make_tifxyz(tmp_path, "outer_shell", scale=(19.997318, 19.996687), step=20.49)
    s = measure_spacing(d, tmp_path, target_spacing=20.0, threshold=0.15)
    fix_metadata_scale(d / "meta.json", s.median_right, s.median_down)

    s2 = measure_spacing(d, tmp_path, target_spacing=20.0, threshold=0.15)
    assert s2.metadata_mismatch is False
    assert s2.metadata_factor == pytest.approx(1.0, abs=1e-6)


# --- CLI end to end, including the compound-case guard ----------------------

def _run_cli(args):
    import subprocess
    import sys as _sys

    script = Path(__file__).resolve().parents[1] / "repair_tifxyz_spacing.py"
    return subprocess.run([_sys.executable, str(script), *args],
                          capture_output=True, text=True)


def test_cli_fix_metadata_end_to_end(tmp_path):
    src = tmp_path / "in"
    out = tmp_path / "out"
    make_tifxyz(src, "outer_shell", scale=(19.997318, 19.996687), step=20.49)
    make_tifxyz(src, "verified_patch", scale=(0.05, 0.05), step=20.0)

    r = _run_cli([str(src), str(out), "--fix-metadata"])
    assert r.returncode == 0, r.stderr
    assert "metadata_factor" not in r.stdout  # sanity: not leaking repr internals
    assert "fix-metadata outer_shell" in r.stdout

    fixed = json.loads((out / "outer_shell" / "meta.json").read_text())
    assert fixed["scale"][0] == pytest.approx(1 / 20.49, rel=1e-3)

    untouched = json.loads((out / "verified_patch" / "meta.json").read_text())
    assert untouched["scale"] == [0.05, 0.05]

    x_src = tifffile.imread(src / "outer_shell" / "x.tif")
    x_out = tifffile.imread(out / "outer_shell" / "x.tif")
    assert np.array_equal(x_src, x_out)


def test_cli_skips_compound_case(tmp_path):
    """A file needing BOTH geometry and metadata repair must not have its
    scale silently rewritten from stale pre-resample measurements."""
    src = tmp_path / "in"
    out = tmp_path / "out"
    make_tifxyz(src, "double_broken", scale=(19.99, 19.99), step=35.0)

    r = _run_cli([str(src), str(out), "--fix-metadata"])
    assert r.returncode == 0, r.stderr
    assert "double_broken" in r.stdout

    result = json.loads((out / "double_broken" / "meta.json").read_text())
    # Existing geometry-repair behaviour restores the ORIGINAL declared scale
    # after resampling -- unrelated to this change, and left as-is.
    assert result["scale"] == [19.99, 19.99]


def test_cli_without_fix_metadata_only_reports(tmp_path):
    """Without --fix-metadata, the file must be reported but not modified."""
    src = tmp_path / "in"
    out = tmp_path / "out"
    make_tifxyz(src, "outer_shell", scale=(19.997318, 19.996687), step=20.49)

    r = _run_cli([str(src), str(out)])
    assert r.returncode == 0, r.stderr
    assert "outer_shell" in r.stdout
    assert "disagrees with" in r.stdout

    result = json.loads((out / "outer_shell" / "meta.json").read_text())
    assert result["scale"] == [19.997318, 19.996687]  # unchanged


def test_dry_run_reports_metadata_mismatch_without_writing(tmp_path):
    src = tmp_path / "in"
    out = tmp_path / "out"
    make_tifxyz(src, "outer_shell", scale=(19.997318, 19.996687), step=20.49)

    r = _run_cli([str(src), str(out), "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "disagrees with" in r.stdout
    assert not out.exists()
