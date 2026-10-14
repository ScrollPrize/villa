import json

import numpy as np

from vesuvius.data.pyramid_occupancy import check_pyramid, expected_coarse_occupancy
from vesuvius.scripts.check_pyramid_occupancy import main


def _write_level(root, path, shape, chunks, present, sep="."):
    d = root / path
    d.mkdir(parents=True)
    (d / ".zarray").write_text(json.dumps({
        "zarr_format": 2, "shape": shape, "chunks": chunks, "dtype": "|u1",
        "compressor": None, "fill_value": 0, "order": "C", "filters": None,
        "dimension_separator": sep,
    }))
    for idx in present:
        f = d.joinpath(*map(str, idx)) if sep == "/" else d / ".".join(map(str, idx))
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")


def _store(tmp_path, l1_present, sep="."):
    root = tmp_path / "s.zarr"
    root.mkdir(parents=True)
    (root / ".zattrs").write_text(json.dumps({"multiscales": [{"datasets": [
        {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}]},
        {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [2, 2, 2]}]},
    ]}]}))
    # level 0: 8^3 voxels, 2^3 chunks -> 4^3 grid; occupy z-chunks 0 and 3
    l0 = [(z, y, x) for z in (0, 3) for y in range(4) for x in range(4)]
    _write_level(root, "0", [8, 8, 8], [2, 2, 2], l0, sep)
    _write_level(root, "1", [4, 4, 4], [2, 2, 2], l1_present, sep)
    return root


FULL_L1 = [(z, y, x) for z in (0, 1) for y in (0, 1) for x in (0, 1)]


def test_expected_coarse_occupancy_factor2():
    fine = np.zeros((4, 4, 4), bool)
    fine[3, 0, 0] = True
    exp = expected_coarse_occupancy(fine, (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2))
    assert exp.sum() == 1 and exp[1, 0, 0]


def test_expected_coarse_occupancy_clips_partial_edge():
    fine = np.zeros((3,), bool)
    fine[2] = True  # fine chunk covering voxels 10..14 of a 13-long axis
    exp = expected_coarse_occupancy(fine, (5,), (4,), (2,), (2.0,))
    assert exp.tolist() == [False, True]


def test_consistent_pyramid_ok(tmp_path):
    (r,) = check_pyramid(str(_store(tmp_path, FULL_L1)))
    assert r.ok and r.expected == 8 and r.missing == 0


def test_missing_coarse_chunk_reported(tmp_path):
    l1 = [c for c in FULL_L1 if c != (1, 1, 0)]
    (r,) = check_pyramid(str(_store(tmp_path, l1)))
    assert r.missing == 1 and r.examples == [(1, 1, 0)]


def test_nested_separator(tmp_path):
    l1 = [c for c in FULL_L1 if c[0] == 0]
    (r,) = check_pyramid(str(_store(tmp_path, l1, sep="/")))
    assert r.missing == 4


def test_empty_coarse_level_all_missing(tmp_path):
    (r,) = check_pyramid(str(_store(tmp_path, [])))
    assert r.missing == r.expected == 8


def test_cli_exit_code(tmp_path, capsys):
    assert main([str(_store(tmp_path / "a", FULL_L1))]) == 0
    assert main([str(_store(tmp_path / "b", FULL_L1[:-1]))]) == 1
    assert "HOLES 1/8" in capsys.readouterr().out
