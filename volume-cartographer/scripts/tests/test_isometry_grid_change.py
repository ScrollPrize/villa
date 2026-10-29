"""A stale `isometry` block (#2279, villa PR #2352 review) must not survive a grid change.

CLI-level regression tests for the writers that change or resynthesize a tifxyz
grid but copy the source `meta.json` (or rewrite it) without re-measuring the
block.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import tifffile


ROOT = Path(__file__).resolve().parents[2]
SEGMENT = ROOT / "core/test/data/segments/20241113070770"
BLOCK = {"version": 3, "edges": 100, "step_vx": [128.0, 128.0]}


def _binary(name):
    path = ROOT / "build" / "bin" / name
    if not path.is_file():
        pytest.skip(f"Build {name} with ninja -C build first")
    return path


def _run(cmd):
    return subprocess.run([str(arg) for arg in cmd], check=True, capture_output=True, text=True)


def _meta(directory):
    return json.loads((Path(directory) / "meta.json").read_text())


@pytest.fixture
def segment_with_block(tmp_path):
    """The real segment fixture, carrying a synthetic `isometry` block."""
    dest = tmp_path / "seg"
    shutil.copytree(SEGMENT, dest)
    meta_path = dest / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["isometry"] = dict(BLOCK)
    meta_path.write_text(json.dumps(meta))
    return dest


def test_vc_straighten_drops_the_block(segment_with_block, tmp_path):
    out = tmp_path / "straight"
    _run([_binary("vc_straighten"), segment_with_block, out, "--overlap-pairs", "0"])
    assert "isometry" not in _meta(out)


def test_vc_transform_geom_drops_the_block_when_it_regrids(segment_with_block, tmp_path):
    # A 2x pre-scale changes the measured spacing, so the grid is resampled back.
    out = tmp_path / "scaled"
    _run([_binary("vc_transform_geom"), "-i", segment_with_block, "-o", out,
          "--scale-before-affine", "2"])
    assert "isometry" not in _meta(out)


def test_vc_transform_geom_rescales_the_block_without_resampling(segment_with_block, tmp_path):
    # --no-resample keeps the grid, so the block stays valid and follows the scale.
    out = tmp_path / "noscale"
    _run([_binary("vc_transform_geom"), "-i", segment_with_block, "-o", out,
          "--scale-before-affine", "2", "--no-resample"])
    iso = _meta(out)["isometry"]
    assert iso["step_vx"][0] == pytest.approx(BLOCK["step_vx"][0] * 2)
    assert iso["step_vx"][1] == pytest.approx(BLOCK["step_vx"][1] * 2)


def test_repair_tifxyz_spacing_drops_the_block(tmp_path):
    pytest.importorskip("cv2")
    src_root = tmp_path / "in"
    seg = src_root / "seg"
    seg.mkdir(parents=True)
    side, spacing = 30, 40.0
    cols, rows = np.meshgrid(np.arange(side), np.arange(side))
    for band, values in (("x", (cols * spacing).astype(np.float32)),
                         ("y", (rows * spacing).astype(np.float32)),
                         ("z", np.full((side, side), 100.0, np.float32))):
        tifffile.imwrite(seg / f"{band}.tif", values)
    # scale 0.05 promises a 20 vx step while the grid is 40 vx, so it gets repaired.
    (seg / "meta.json").write_text(json.dumps(
        {"uuid": "repair-fixture", "scale": [0.05, 0.05], "isometry": dict(BLOCK)}))
    out_root = tmp_path / "out"
    _run([sys.executable, ROOT / "scripts" / "repair_tifxyz_spacing.py", src_root, out_root])
    assert "isometry" not in _meta(out_root / "seg")
