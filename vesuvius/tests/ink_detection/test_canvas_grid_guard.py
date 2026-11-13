from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import warnings

import zarr

from vesuvius.ink_detection.data.dataset import InkDataset


def _canvas(path: Path, shape: tuple[int, ...]):
    return zarr.open(str(path), mode="w", shape=shape, dtype="uint8")


def _check(tmp_path: Path, label_shape, grid_shape):
    stub = SimpleNamespace(_canvas_checked=set())
    segment = SimpleNamespace(segment_dir=tmp_path, segment_name="w018")
    image = _canvas(tmp_path / "image.zarr", (3, 100, 120))
    labels = _canvas(tmp_path / "labels.zarr", label_shape)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            InkDataset._check_canvas_grid(
                stub, segment, image, {"inklabels": labels}, grid_shape
            )
    return [str(w.message) for w in caught]


def test_smaller_label_canvas_warns_once_with_centred_offset(tmp_path):
    messages = _check(tmp_path, (90, 110), (100, 120))
    assert len(messages) == 1
    assert "w018" in messages[0]
    assert "(5, 5)" in messages[0]


def test_matching_label_canvas_is_silent(tmp_path):
    assert _check(tmp_path, (100, 120), (100, 120)) == []
