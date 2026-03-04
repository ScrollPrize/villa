import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile

from vesuvius.tifxyz import read_tifxyz


def _write_segment(path: Path, shape: tuple[int, int], scale: tuple[float, float] = (1.0, 1.0)) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    h, w = shape
    y_grid, x_grid = np.mgrid[:h, :w]
    z_grid = np.ones((h, w), dtype=np.float32)

    tifffile.imwrite(str(path / "x.tif"), x_grid.astype(np.float32))
    tifffile.imwrite(str(path / "y.tif"), y_grid.astype(np.float32))
    tifffile.imwrite(str(path / "z.tif"), z_grid)

    scale_y, scale_x = scale
    meta = {
        "uuid": "segment",
        "scale": [float(scale_x), float(scale_y)],
        "bbox": None,
        "area": None,
    }
    (path / "meta.json").write_text(json.dumps(meta))
    return path


def _write_label(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to write test image: {path}")


def test_list_labels_discovers_paths_shapes_and_status(tmp_path: Path) -> None:
    segment_dir = _write_segment(tmp_path / "segment", (6, 8))
    _write_label(segment_dir / "a_ink.tif", np.full((6, 8), 7, dtype=np.uint8))
    _write_label(segment_dir / "b_surface.png", np.full((6, 8), 9, dtype=np.uint8))
    _write_label(segment_dir / "c_ink.jpg", np.full((5, 8), 3, dtype=np.uint8))

    segment = read_tifxyz(segment_dir)
    labels = segment.list_labels()

    assert [label["filename"] for label in labels] == [
        "a_ink.tif",
        "b_surface.png",
        "c_ink.jpg",
    ]

    c_ink = labels[2]
    assert c_ink["name"] == "ink"
    assert c_ink["path"] == segment_dir / "c_ink.jpg"
    assert c_ink["shape"] == (5, 8)
    assert c_ink["matches_stored_shape"] is False
    assert "Shape mismatch" in str(c_ink["error"])


def test_load_label_by_index_filename_and_suffix(tmp_path: Path) -> None:
    segment_dir = _write_segment(tmp_path / "segment", (4, 5))
    ink_u16 = (np.arange(20, dtype=np.uint16).reshape(4, 5) * 200)
    surface_u8 = np.full((4, 5), 42, dtype=np.uint8)
    _write_label(segment_dir / "foo_ink.tif", ink_u16)
    _write_label(segment_dir / "bar_surface.png", surface_u8)

    segment = read_tifxyz(segment_dir)
    labels = segment.list_labels()
    surface_idx = next(
        label["index"] for label in labels if label["filename"] == "bar_surface.png"
    )

    loaded_by_index = segment.load_label(surface_idx)
    loaded_by_filename = segment.load_label("bar_surface.png")
    loaded_by_suffix = segment.load_label("ink")

    np.testing.assert_array_equal(loaded_by_index, loaded_by_filename)
    assert loaded_by_suffix.dtype == np.uint8
    assert loaded_by_suffix.shape == (4, 5)
    assert int(loaded_by_suffix.min()) == 0
    assert int(loaded_by_suffix.max()) == 255


def test_load_label_raises_for_ambiguous_suffix(tmp_path: Path) -> None:
    segment_dir = _write_segment(tmp_path / "segment", (3, 3))
    _write_label(segment_dir / "a_ink.tif", np.ones((3, 3), dtype=np.uint8))
    _write_label(segment_dir / "b_ink.jpg", np.ones((3, 3), dtype=np.uint8))

    segment = read_tifxyz(segment_dir)
    with pytest.raises(ValueError, match="Ambiguous label suffix"):
        segment.load_label("ink")


def test_load_label_uses_stored_shape_even_in_full_resolution_mode(tmp_path: Path) -> None:
    segment_dir = _write_segment(tmp_path / "segment", (6, 7), scale=(2.0, 2.0))
    _write_label(segment_dir / "good_ink.tif", np.ones((6, 7), dtype=np.uint8))
    _write_label(segment_dir / "bad_surface.png", np.ones((5, 7), dtype=np.uint8))

    segment = read_tifxyz(segment_dir).use_full_resolution()
    loaded = segment.load_label("ink")
    assert loaded.shape == (6, 7)

    with pytest.raises(ValueError, match="Shape mismatch"):
        segment.load_label("surface")


def test_load_label_rejects_non_grayscale(tmp_path: Path) -> None:
    segment_dir = _write_segment(tmp_path / "segment", (5, 6))
    rgb = np.zeros((5, 6, 3), dtype=np.uint8)
    rgb[..., 1] = 255
    _write_label(segment_dir / "foo_rgb.png", rgb)

    segment = read_tifxyz(segment_dir)
    with pytest.raises(ValueError, match="2D grayscale"):
        segment.load_label("rgb")


def test_filename_match_precedes_suffix_match(tmp_path: Path) -> None:
    segment_dir = _write_segment(tmp_path / "segment", (4, 4))
    _write_label(segment_dir / "ink.jpg", np.full((4, 4), 11, dtype=np.uint8))
    _write_label(segment_dir / "foo_ink.tif", np.full((4, 4), 22, dtype=np.uint8))

    segment = read_tifxyz(segment_dir)
    loaded = segment.load_label("ink.jpg")
    assert int(loaded[0, 0]) == 11
