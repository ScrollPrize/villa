import json
import sqlite3

import numpy as np
import pytest
import torch
import zarr

from vesuvius.afv_spline_generator import cli


def bright_voxels_are_vertical_fibers():
    """Logits favouring the vertical-fiber class on bright voxels, background elsewhere."""
    network = torch.nn.Conv3d(1, 4, 1)
    with torch.no_grad():
        network.weight.copy_(torch.tensor([-4.0, 4.0, 0.0, 0.0]).view(4, 1, 1, 1, 1))
        network.bias.copy_(torch.tensor([0.0, 0.0, -8.0, -8.0]))
    return network.eval()


@pytest.fixture
def volume(tmp_path):
    """An OME-Zarr group whose array 0 holds a bright fiber along x at y=30, z=20."""
    path = tmp_path / "volume.zarr"
    zarr.open_group(str(path), mode="w")
    array = zarr.open_array(str(path / "0"), mode="w", shape=(48, 64, 96), chunks=(16, 16, 16), dtype="u1")
    ct = np.full(array.shape, 20, dtype=np.uint8)
    z, y = np.ogrid[:48, :64]
    ct[((z - 20) ** 2 + (y - 30) ** 2 <= 6)[:, :, None].repeat(96, axis=2) & (np.arange(96) >= 10) & (np.arange(96) < 80)] = 200
    array[:] = ct
    return path


def run(monkeypatch, capsys, volume, output, *extra):
    monkeypatch.setattr(cli, "resolve_model", lambda model: (volume, {"model": model}))
    monkeypatch.setattr(cli, "load_network", lambda folder, device: (bright_voxels_are_vertical_fibers(), (16, 16, 16)))
    code = cli.main([
        "--volume", str(volume),
        "--origin", "4", "10", "8",
        "--size", "88", "40", "30",
        "--output", str(output),
        "--coordinate-space", "PHercTest/20250101000000",
        "--native-scale", "2",
        "--voxel-size", "4.5",
        "--device", "cpu",
        "--progress", "json",
        *extra,
    ])
    return code, [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_generates_a_volume_in_native_coordinates(monkeypatch, capsys, volume, tmp_path):
    output = tmp_path / "fibers.afv"
    code, events = run(monkeypatch, capsys, volume, output)
    assert code == 0
    done = events[-1]
    assert done["event"] == "done" and done["output"] == str(output) and done["fibers"] == 1
    fractions = [e["fraction"] for e in events if e["event"] == "progress"]
    assert fractions == sorted(fractions) and {e["stage"] for e in events if e["event"] == "progress"} == set(cli.Reporter.STAGES)

    db = sqlite3.connect(output)
    metadata = {key: json.loads(value) for key, value in db.execute("SELECT key, value FROM metadata")}
    assert metadata["frame"]["vc_open_data_coordinate_space"] == "PHercTest/20250101000000"
    assert metadata["root"]["vc_open_data_source_original_resolution"] == 4.5
    # The native shape is not known exactly from a downsampled volume.
    assert "coordinate_base_shape_zyx" not in metadata["root"]
    assert metadata["generator"]["origin_xyz"] == [4, 10, 8] and metadata["generator"]["mirror"] is False
    family, length, annotation = db.execute("SELECT family, length, annotation FROM fibers").fetchone()
    points = np.concatenate([np.frombuffer(blob, "<f8").reshape(-1, 3) for (blob,) in db.execute("SELECT points FROM blocks ORDER BY first_segment")])
    assert family == "V"
    # The fiber spans x in [10, 80) at y=30, z=20 of the volume, twice that in native voxels.
    assert np.abs(points[:, 1:] - [60, 40]).max() < 1
    assert 2 * 10 <= points[:, 0].min() < 2 * 14 and 2 * 76 < points[:, 0].max() <= 2 * 80
    assert json.loads(annotation)["length_mm"] == pytest.approx(length * 0.0045)


def test_blocks_are_stitched_and_previewed(monkeypatch, capsys, volume, tmp_path):
    output = tmp_path / "fibers.afv"
    previews = tmp_path / "previews"
    previews.mkdir()
    code, events = run(monkeypatch, capsys, volume, output, "--block-size", "48", "--preview-dir", str(previews))
    assert code == 0
    plan = next(e for e in events if e["event"] == "plan")
    assert plan["blocks"] == [{"origin": [4, 10, 8], "size": [48, 40, 30]}, {"origin": [52, 10, 8], "size": [40, 40, 30]}]
    for index in (0, 1):
        states = [e["state"] for e in events if e["event"] == "block" and e["index"] == index]
        assert states[:5] == ["reading", "predicting", "splines", "stitching", "stitched"] and states[-1] == "done"
    shown = [e for e in events if e["event"] == "preview"]
    assert [e["path"] for e in shown] == [str(previews / "preview-0001.afv"), str(previews / "preview-0002.afv")]
    assert all(e["fibers"] == 1 for e in shown)
    # Both blocks see the fiber; it is written once.
    assert events[-1]["fibers"] == 1


def test_zone_is_cut_into_blocks():
    assert cli.zone_blocks([4, 10, 8], [88, 40, 30], 48) == [([4, 10, 8], [48, 40, 30]), ([52, 10, 8], [40, 40, 30])]
    assert len(cli.zone_blocks([0, 0, 0], [1024, 1024, 512], 512)) == 4
    assert cli.zone_blocks([0, 0, 0], [1024, 1024, 512], 512)[1] == ([512, 0, 0], [512, 512, 512])


@pytest.mark.parametrize("spilled", [False, True])
def test_region_reader_matches_the_volume(volume, tmp_path, spilled):
    array = cli.open_volume(str(volume), 0)
    read = cli.RegionReader(array, budget=3 * 16**3, spill=tmp_path if spilled else None)
    for _ in range(2):
        for low, high in (([0, 0, 0], [96, 64, 48]), ([5, 17, 31], [70, 40, 33]), ([90, 60, 40], [96, 64, 48])):
            assert np.array_equal(read(low, high), array[low[2]:high[2], low[1]:high[1], low[0]:high[0]])
    assert len(read.cache) <= 3
    assert len(list(tmp_path.glob("*.npy"))) == (len(list(np.ndindex(*np.ceil(np.asarray(array.shape) / array.chunks).astype(int)))) if spilled else 0)


def test_reports_errors_as_json_and_writes_nothing(monkeypatch, capsys, volume, tmp_path):
    output = tmp_path / "fibers.afv"
    code, events = run(monkeypatch, capsys, volume, output, "--level", "1")
    assert code == 1 and events[-1] == {"event": "error", "message": "The volume has no array 1"}
    assert not output.exists()


@pytest.mark.parametrize("flag, mirror", [((), False), (("--mirror",), True), (("--no-mirror",), False)])
def test_mirroring_is_off_unless_asked(flag, mirror):
    required = ["--volume", "v.zarr", "--origin", "0", "0", "0", "--size", "1", "1", "1", "--output", "f.afv", "--coordinate-space", "S/1"]
    assert cli.build_parser().parse_args([*required, *flag]).mirror is mirror


def test_zone_must_be_inside_the_volume():
    assert cli.zone_slices((10, 20, 30), (1, 2, 3), (4, 5, 6)) == (slice(3, 9), slice(2, 7), slice(1, 5))
    with pytest.raises(ValueError, match="along x"):
        cli.zone_slices((10, 20, 30), (28, 0, 0), (4, 5, 6))


def test_model_paths_are_not_mistaken_for_repositories(tmp_path):
    assert cli.resolve_model(str(tmp_path))[0] == tmp_path
    with pytest.raises(FileNotFoundError):
        cli.resolve_model(str(tmp_path / "missing" / "model"))
