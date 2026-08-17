from vesuvius.data import utils


def test_open_zarr_omits_storage_options_for_local_path(monkeypatch, tmp_path):
    captured = {}

    def fake_open(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(utils.zarr, "open", fake_open)

    utils.open_zarr(str(tmp_path / "local.zarr"), storage_options={})

    assert captured["path"] == str(tmp_path / "local.zarr")
    assert "storage_options" not in captured


def test_open_zarr_forwards_storage_options_for_remote_uri(monkeypatch):
    captured = {}

    def fake_open(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(utils.zarr, "open", fake_open)

    utils.open_zarr("https://example.com/volume.zarr", storage_options={"token": "secret"})

    assert captured["path"] == "https://example.com/volume.zarr"
    assert captured["storage_options"] == {"token": "secret"}
