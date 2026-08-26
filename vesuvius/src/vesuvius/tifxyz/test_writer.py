import json
from pathlib import Path

import numpy as np

from vesuvius.tifxyz import read_tifxyz, write_tifxyz
from vesuvius.tifxyz.types import Tifxyz
from vesuvius.tifxyz.writer import TifxyzWriter


def _fresh_surface() -> Tifxyz:
    h, w = 4, 5
    y_grid, x_grid = np.mgrid[:h, :w]
    return Tifxyz(
        _x=(10.0 + x_grid).astype(np.float32),
        _y=(20.0 + y_grid).astype(np.float32),
        _z=np.full((h, w), 30.0, dtype=np.float32),
        uuid="segment",
    )


def _load_mutate(tmp_path: Path) -> Tifxyz:
    """Write a surface, read it back, then edit its coordinates in place."""
    write_tifxyz(tmp_path / "segment", _fresh_surface())
    surface = read_tifxyz(tmp_path / "segment")
    assert surface.bbox == (10.0, 20.0, 30.0, 14.0, 23.0, 30.0)
    surface._x[:] += 100.0  # e.g. translate the surface after loading
    return surface


def test_write_metadata_recomputes_bbox_after_inplace_edit(tmp_path: Path) -> None:
    surface = _load_mutate(tmp_path)

    TifxyzWriter(tmp_path / "segment", overwrite=True).write_metadata(surface)

    meta = json.loads((tmp_path / "segment" / "meta.json").read_text())
    assert meta["bbox"] == [[110.0, 20.0, 30.0], [114.0, 23.0, 30.0]]


def test_write_metadata_repairs_a_stale_stored_bbox(tmp_path: Path) -> None:
    """A surface loaded with a wrong stored bbox is written out corrected."""
    write_tifxyz(tmp_path / "segment", _fresh_surface())
    meta_path = tmp_path / "segment" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["bbox"] = [[-0.25, -0.25, -0.25], [14.0, 23.0, 30.0]]
    meta_path.write_text(json.dumps(meta))

    surface = read_tifxyz(tmp_path / "segment")
    TifxyzWriter(tmp_path / "segment", overwrite=True).write_metadata(surface)

    assert json.loads(meta_path.read_text())["bbox"] == [
        [10.0, 20.0, 30.0], [14.0, 23.0, 30.0]
    ]


def test_write_metadata_opt_out_keeps_stored_bbox(tmp_path: Path) -> None:
    surface = _load_mutate(tmp_path)

    TifxyzWriter(tmp_path / "segment", overwrite=True).write_metadata(
        surface, recompute_bbox=False
    )

    meta = json.loads((tmp_path / "segment" / "meta.json").read_text())
    assert meta["bbox"] == [[10.0, 20.0, 30.0], [14.0, 23.0, 30.0]]


def test_explicit_bbox_argument_still_wins(tmp_path: Path) -> None:
    surface = _load_mutate(tmp_path)

    TifxyzWriter(tmp_path / "segment", overwrite=True).write_metadata(
        surface, bbox=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    )

    meta = json.loads((tmp_path / "segment" / "meta.json").read_text())
    assert meta["bbox"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
