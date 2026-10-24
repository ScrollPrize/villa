"""meta.json uuid identity for inference tifxyz outputs.

A new output surface gets a fresh uuid4; --overwrite-input-surface rewrites
the input in place and must keep its existing uuid (never the directory name).
"""
import json
import uuid
from types import SimpleNamespace

import numpy as np

from vesuvius.neural_tracing.inference.common import _save_merged_surface_tifxyz

INPUT_UUID = "0f8fad5b-d9cb-469f-a165-70867728950e"


def _make_input(tmp_path):
    surface = tmp_path / "segment_name.tifxyz"
    surface.mkdir()
    (surface / "meta.json").write_text(json.dumps(
        {"uuid": INPUT_UUID, "scale": [0.1, 0.1], "format": "tifxyz", "type": "seg"}))
    return surface


def _args(surface, *, overwrite, out_dir=None):
    return SimpleNamespace(
        volume_scale=0, tifxyz_path=str(surface), tifxyz_step_size=None,
        tifxyz_voxel_size_um=8.0, volume_path=str(surface.parent),
        overwrite_input_surface=overwrite, tifxyz_out_dir=out_dir,
        grow_direction="up", extrapolation_method="test", agg_extrap_lines=None)


def _merged():
    rows, cols = np.mgrid[:4, :4].astype(np.float32)
    zyxs = np.stack([np.full_like(rows, 50.0), rows * 10 + 100, cols * 10 + 200], axis=-1)
    return {"merged_zyxs": zyxs, "merged_valid": np.ones((4, 4), dtype=bool)}


def test_overwrite_input_surface_preserves_existing_uuid(tmp_path):
    surface = _make_input(tmp_path)
    output = _save_merged_surface_tifxyz(
        _args(surface, overwrite=True), _merged(), None, None, {})
    assert output == str(surface)
    meta = json.loads((surface / "meta.json").read_text())
    assert meta["uuid"] == INPUT_UUID


def test_new_output_surface_gets_fresh_uuid(tmp_path):
    surface = _make_input(tmp_path)
    out_dir = tmp_path / "out"
    output = _save_merged_surface_tifxyz(
        _args(surface, overwrite=False, out_dir=str(out_dir)), _merged(), None, None, {})
    meta = json.loads((out_dir / output.split("/")[-1] / "meta.json").read_text())
    assert meta["uuid"] != INPUT_UUID
    assert uuid.UUID(meta["uuid"]).version == 4
    assert meta["uuid"] != output.split("/")[-1]
