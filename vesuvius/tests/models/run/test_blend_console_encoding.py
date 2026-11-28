"""Blending's progress output must survive a non-UTF-8 stdout.

On Windows, Python encodes a redirected or piped stdout (``vesuvius.blend_logits ... > blend.log``,
a job scheduler capturing output) with the ANSI code page, cp1252, which has no check mark.
"""
from __future__ import annotations

import io
import json
import sys

import numpy as np

from vesuvius.models.run import blending


class _Store:
    """Stands in for ObjectStore: a part whose non-empty patch cache says both patches are present."""

    def __init__(self, path):
        self.path = path

    def read(self, key):
        if key == ".non_empty_patch_idxs.json":
            return json.dumps({"num_patches": 2, "non_empty_indices": [0, 1]})
        return None

    def write(self, key, data):
        pass


class _Coords:
    """Stands in for a coordinates_part zarr: two patch origins, no cached bbox."""

    attrs: dict = {}

    def __getitem__(self, key):
        return np.array([[0, 0, 0], [8, 8, 8]], dtype=np.int32)


def test_spatial_grid_verbose_output_encodes_as_cp1252(monkeypatch) -> None:
    monkeypatch.setattr(blending, "ObjectStore", _Store)
    monkeypatch.setattr(blending, "open_zarr", lambda path: _Coords())
    raw = io.BytesIO()
    stdout = io.TextIOWrapper(raw, encoding="cp1252")  # what Windows gives a redirected stdout
    monkeypatch.setattr(sys, "stdout", stdout)

    grid = blending.SpatialPatchGrid(patch_size=(16, 16, 16))
    grid.build("parts", {0: {"logits": "logits_part_0.zarr", "coordinates": "coordinates_part_0.zarr"}},
               [0], (16, 16, 16), verbose=True)

    stdout.flush()
    assert b"Grid complete: 1 cells, 2 non-empty patches" in raw.getvalue()
