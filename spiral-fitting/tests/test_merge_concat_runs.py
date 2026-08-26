import json
import sys
from pathlib import Path

import numpy as np

SPIRAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPIRAL_DIR))

from merge_concat_runs import save_tifxyz  # noqa: E402
from tifxyz import save_tifxyz as save_tifxyz_canonical  # noqa: E402


def grid():
    """3x3 grid with x/y/z in distinct ranges, so an axis swap is visible."""
    pts = np.zeros((3, 3, 3), np.float32)
    pts[..., 0] = 100 + np.arange(3)[None, :]                          # x
    pts[..., 1] = 200 + np.arange(3)[:, None]                          # y
    pts[..., 2] = 300 + np.arange(3)[:, None] + np.arange(3)[None, :]  # z
    return pts


def written_bbox(out_dir):
    return json.loads((Path(out_dir) / "meta.json").read_text())["bbox"]


def test_bbox_masks_invalid_vertices_and_is_xyz_ordered(tmp_path):
    pts = grid()
    pts[0, 0] = -1.0
    save_tifxyz(pts, tmp_path / "merged", "w000", 20.0, 7.91, "test")

    valid = np.any(pts != -1, axis=-1)
    assert written_bbox(tmp_path / "merged") == [
        pts[valid].min(axis=0).tolist(),
        pts[valid].max(axis=0).tolist(),
    ]


def test_bbox_matches_canonical_spiral_writer(tmp_path):
    pts = grid()
    pts[2, 2] = -1.0
    save_tifxyz(pts, tmp_path / "merged", "w000", 20.0, 7.91, "test")
    # the canonical writer takes z,y,x-ordered input
    save_tifxyz_canonical(pts[..., ::-1], str(tmp_path), "ref", 20.0, 7.91, "test")

    assert written_bbox(tmp_path / "merged") == written_bbox(tmp_path / "ref")


def test_all_invalid_grid_falls_back_to_sentinel_bbox(tmp_path):
    save_tifxyz(np.full((3, 3, 3), -1.0, np.float32), tmp_path / "merged",
                "w000", 20.0, 7.91, "test")

    assert written_bbox(tmp_path / "merged") == [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]
