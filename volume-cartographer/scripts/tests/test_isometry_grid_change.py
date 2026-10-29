"""A stale `isometry` block (#2279, villa PR #2352 review) must not survive a grid change.

CLI-level regression test for a writer that changes a tifxyz grid but copies the
source `meta.json` without re-measuring the block.
"""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SEGMENT = ROOT / "core/test/data/segments/20241113070770"
BLOCK = {"version": 3, "edges": 100, "step_vx": [128.0, 128.0]}


def _binary(name):
    path = ROOT / "build" / "bin" / name
    if not path.is_file():
        pytest.skip(f"Build {name} with ninja -C build first")
    return path


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
    subprocess.run([str(_binary("vc_straighten")), str(segment_with_block), str(out),
                    "--overlap-pairs", "0"], check=True, capture_output=True, text=True)
    assert "isometry" not in _meta(out)
