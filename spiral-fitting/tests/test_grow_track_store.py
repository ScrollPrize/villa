"""Tests for grow_track_graph.py store mode: patches become stored surfaces."""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grow_track_graph as ggt  # noqa: E402

vc_object_store = pytest.importorskip("vc_object_store")


def test_grown_patches_are_stored_surfaces_found_by_seed(tmp_path):
    store = tmp_path / "objects.vcobj"
    vc_object_store.ObjectStore.create(str(store), author="pytest").close()
    tracks = tmp_path / "tracks.vctracks"
    cols, rows = np.meshgrid(np.arange(4.0), np.arange(3.0))
    grid = np.stack((cols, rows, np.full((3, 4), 50.0)), -1) * 20.0
    bordered, meta = ggt.output_surface(grid, {"seed_track_row": 7, "tracks_path": str(tracks.resolve())})
    args = Namespace(store=store, volume_uuid="vol", tag=["paths"])
    uid = ggt.store_surface(args, 7, bordered, meta, (0.05, 0.05))

    points, valid, scale, record = vc_object_store.surfaces.read_surface(store, uid, volume="vol")
    assert valid.sum() == 12 and not valid[0].any() and scale == (0.05, 0.05)
    np.testing.assert_array_equal(points[1:-1, 1:-1], grid.astype(np.float32))
    assert record["tags"] == ["paths"] and record["meta"]["provenance"]["tool"] == "grow_track_graph"
    assert record["meta"]["area_cm2"] == meta["area_cm2"]

    assert ggt.stored_seed_surfaces(store, tracks) == {7: [uid]}
    assert ggt.stored_seed_surfaces(store, tmp_path / "other.vctracks") == {}
    ggt.retire_surfaces(store, [uid])
    assert ggt.stored_seed_surfaces(store, tracks) == {}


@pytest.mark.parametrize("argv", [
    ["t", "c"],
    ["t", "c", "out", "--store", "s.vcobj", "--volume-uuid", "v"],
    ["t", "c", "--store", "s.vcobj"],
    ["t", "c", "out", "--tag", "paths"],
])
def test_store_mode_arguments(argv):
    with pytest.raises(SystemExit):
        ggt.main(argv + ["--seeds", "1"])
