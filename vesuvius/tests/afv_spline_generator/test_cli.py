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
    assert metadata["generator"]["origin_xyz"] == [4, 10, 8] and metadata["generator"]["mirror"] is True
    family, length, annotation = db.execute("SELECT family, length, annotation FROM fibers").fetchone()
    points = np.concatenate([np.frombuffer(blob, "<f8").reshape(-1, 3) for (blob,) in db.execute("SELECT points FROM blocks ORDER BY first_segment")])
    assert family == "V"
    # The fiber spans x in [10, 80) at y=30, z=20 of the volume, twice that in native voxels.
    assert np.abs(points[:, 1:] - [60, 40]).max() < 1
    assert 2 * 10 <= points[:, 0].min() < 2 * 14 and 2 * 76 < points[:, 0].max() <= 2 * 80
    assert json.loads(annotation)["length_mm"] == pytest.approx(length * 0.0045)


def test_reports_errors_as_json_and_writes_nothing(monkeypatch, capsys, volume, tmp_path):
    output = tmp_path / "fibers.afv"
    code, events = run(monkeypatch, capsys, volume, output, "--level", "1")
    assert code == 1 and events[-1] == {"event": "error", "message": "The volume has no array 1"}
    assert not output.exists()


def test_zone_must_be_inside_the_volume():
    assert cli.zone_slices((10, 20, 30), (1, 2, 3), (4, 5, 6)) == (slice(3, 9), slice(2, 7), slice(1, 5))
    with pytest.raises(ValueError, match="along x"):
        cli.zone_slices((10, 20, 30), (28, 0, 0), (4, 5, 6))


def test_model_paths_are_not_mistaken_for_repositories(tmp_path):
    assert cli.resolve_model(str(tmp_path))[0] == tmp_path
    with pytest.raises(FileNotFoundError):
        cli.resolve_model(str(tmp_path / "missing" / "model"))
