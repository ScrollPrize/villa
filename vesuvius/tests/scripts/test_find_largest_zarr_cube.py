import json
import math

import numpy as np
import vesuvius.data.zarr_chunk_index as chunk_index
from vesuvius.scripts.find_largest_zarr_cube import (
    find_largest_encoded_cube,
    main,
)


def _brute_force_cube(occupancy, chunks, shape):
    side_unit = math.lcm(*chunks)
    for side in range((min(shape) // side_unit) * side_unit, 0, -side_unit):
        window = tuple(side // chunk for chunk in chunks)
        start_counts = tuple(
            min(grid - width + 1, (logical - side) // chunk + 1)
            for grid, width, logical, chunk in zip(occupancy.shape, window, shape, chunks)
        )
        for z in range(start_counts[0]):
            for y in range(start_counts[1]):
                for x in range(start_counts[2]):
                    if occupancy[
                        z : z + window[0], y : y + window[1], x : x + window[2]
                    ].all():
                        return side, (z, y, x)
    return None


def test_finds_largest_cube_and_uses_lexicographic_tie_break() -> None:
    occupancy = np.zeros((5, 5, 5), dtype=bool)
    occupancy[0:2, 2:4, 1:3] = True
    occupancy[2:4, 0:2, 0:2] = True

    cube = find_largest_encoded_cube(
        occupancy,
        chunks_zyx=(4, 4, 4),
        shape_zyx=(20, 20, 20),
        workers=4,
    )

    assert cube is not None
    assert cube.side_voxels == 8
    assert cube.chunk_start_zyx == (0, 2, 1)
    assert cube.chunk_stop_zyx == (2, 4, 3)
    assert cube.voxel_start_zyx == (0, 8, 4)
    assert cube.voxel_stop_zyx == (8, 16, 12)


def test_supports_anisotropic_chunks() -> None:
    occupancy = np.zeros((3, 4, 4), dtype=bool)
    occupancy[1, 1:3, 2:4] = True

    cube = find_largest_encoded_cube(
        occupancy,
        chunks_zyx=(4, 2, 2),
        shape_zyx=(12, 8, 8),
        workers=2,
    )

    assert cube is not None
    assert cube.side_voxels == 4
    assert cube.chunk_start_zyx == (1, 1, 2)
    assert cube.chunk_stop_zyx == (2, 3, 4)
    assert cube.voxel_start_zyx == (4, 2, 4)
    assert cube.voxel_stop_zyx == (8, 6, 8)


def test_returns_none_for_empty_occupancy() -> None:
    result = find_largest_encoded_cube(
        np.zeros((2, 2, 2), dtype=bool),
        chunks_zyx=(2, 2, 2),
        shape_zyx=(4, 4, 4),
    )
    assert result is None


def test_matches_brute_force_for_random_occupancy() -> None:
    rng = np.random.default_rng(20260825)
    for chunks, shape in [((2, 2, 2), (8, 10, 12)), ((4, 2, 2), (12, 8, 10))]:
        grid = tuple((size + chunk - 1) // chunk for size, chunk in zip(shape, chunks))
        for _ in range(20):
            occupancy = rng.random(grid) < 0.65
            expected = _brute_force_cube(occupancy, chunks, shape)
            actual = find_largest_encoded_cube(
                occupancy, chunks, shape, workers=4
            )
            if expected is None:
                assert actual is None
            else:
                assert actual is not None
                assert (actual.side_voxels, actual.chunk_start_zyx) == expected


def test_cli_accepts_ome_array_group_and_emits_json(tmp_path, capsys) -> None:
    array = tmp_path / "volume.zarr" / "0"
    array.mkdir(parents=True)
    (tmp_path / "volume.zarr" / ".zgroup").write_text(
        json.dumps({"zarr_format": 2}), encoding="ascii"
    )
    (array / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": [8, 8, 8],
                "chunks": [2, 2, 2],
                "dtype": "|u1",
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "dimension_separator": ".",
            }
        ),
        encoding="ascii",
    )
    for z in range(1, 3):
        for y in range(2):
            for x in range(2):
                (array / f"{z}.{y}.{x}").write_bytes(b"encoded")

    status = main([str(array), "--workers", "2", "--json"])

    captured = capsys.readouterr()
    assert status == 0
    output = json.loads(captured.out)
    assert output["zarr"] == str(array)
    assert output["volume_shape_zyx"] == [8, 8, 8]
    assert output["cube"]["side_voxels"] == 4
    assert output["cube"]["voxel_start_zyx"] == [2, 0, 0]
    assert output["cube"]["voxel_stop_zyx"] == [6, 4, 4]
    assert "chunk_start_zyx" not in output["cube"]
    assert not (array / ".chunk_occupancy.npz").exists()


def test_cli_resolves_first_advertised_ome_dataset(tmp_path, capsys) -> None:
    root = tmp_path / "volume.zarr"
    array = root / "0"
    array.mkdir(parents=True)
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}), encoding="ascii")
    (root / ".zattrs").write_text(
        json.dumps(
            {
                "multiscales": [
                    {"datasets": [{"path": "0"}, {"path": "1"}]}
                ]
            }
        ),
        encoding="ascii",
    )
    (array / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": [4, 4, 4],
                "chunks": [2, 2, 2],
                "dtype": "|u1",
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "dimension_separator": ".",
            }
        ),
        encoding="ascii",
    )
    (array / "0.0.0").write_bytes(b"encoded")

    status = main([str(root), "--json"])

    captured = capsys.readouterr()
    assert status == 0
    output = json.loads(captured.out)
    assert output["zarr"] == str(root)
    assert output["cube"]["side_voxels"] == 2


def test_cli_intersects_fiberlet_arrays_and_reports_base_voxels(tmp_path, capsys) -> None:
    root = tmp_path / "fiberlets.zarr"
    root.mkdir()
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}), encoding="ascii")
    (root / ".zattrs").write_text(
        json.dumps(
            {
                "vc_format": "fiberlet_dataset",
                "chunk_grid_shape_zyx": [3, 3, 3],
                "coordinate_origin_zyx": [1, 2, 3],
                "coordinate_units_per_chunk_zyx": [16, 16, 16],
                "spatial_chunk_side_base": 64,
                "prediction_to_base": 2.0,
                "processing": {
                    "grid": {"shape_zyx": [80, 90, 100]},
                    "layout": {"arrays": ["anchors", "prefix", "routes"]},
                },
            }
        ),
        encoding="ascii",
    )
    array_metadata = {
        "zarr_format": 2,
        "shape": [3, 3, 3],
        "chunks": [1, 1, 1],
        "dtype": "|O",
        "compressor": None,
        "fill_value": None,
        "order": "C",
        "filters": None,
        "dimension_separator": ".",
    }
    for name in ("anchors", "prefix", "routes"):
        array = root / name
        array.mkdir()
        (array / ".zarray").write_text(json.dumps(array_metadata), encoding="ascii")
        for z in range(2):
            for y in range(2):
                for x in range(2):
                    (array / f"{z}.{y}.{x}").write_bytes(b"encoded")
    # This anchor-only chunk must not count as an encoded combined chunk.
    (root / "anchors" / "2.2.2").write_bytes(b"anchor only")

    status = main([str(root), "--json", "--workers", "4"])

    captured = capsys.readouterr()
    assert status == 0
    output = json.loads(captured.out)
    assert output["volume_shape_zyx"] == [160, 180, 200]
    assert output["cube"] == {
        "side_voxels": 128,
        "voxel_start_zyx": [4, 8, 12],
        "voxel_stop_zyx": [132, 136, 140],
    }


def test_flat_local_store_is_scanned_once(tmp_path, monkeypatch) -> None:
    array = tmp_path / "volume.zarr"
    array.mkdir()
    (array / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": [8, 8, 8],
                "chunks": [2, 2, 2],
                "dtype": "|u1",
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "dimension_separator": ".",
            }
        ),
        encoding="ascii",
    )
    (array / "3.2.1").write_bytes(b"encoded")

    calls = []
    original = chunk_index._list_chunks_local

    def record_call(array_url: str, sub_prefix: str, sep: str):
        calls.append((sub_prefix, sep))
        return original(array_url, sub_prefix, sep)

    monkeypatch.setattr(chunk_index, "_list_chunks_local", record_call)
    occupancy = chunk_index.build_chunk_occupancy(
        str(array),
        chunks=(2, 2, 2),
        shape=(8, 8, 8),
        use_cache=False,
        workers=4,
    )

    assert occupancy is not None
    assert occupancy[3, 2, 1]
    assert calls == [("", ".")]
