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


def test_open_zarr_translates_creation_keywords_for_zarr2(monkeypatch, tmp_path):
    captured = {}

    def fake_open(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(utils.zarr, "__version__", "2.18.7")
    monkeypatch.setattr(utils.zarr, "open", fake_open)

    utils.open_zarr(
        str(tmp_path / "output.zarr"),
        mode="w",
        shape=(1,),
        dtype="u1",
        zarr_format=2,
        config={"write_empty_chunks": False},
    )

    assert captured["zarr_version"] == 2
    assert captured["write_empty_chunks"] is False
    assert "zarr_format" not in captured
    assert "config" not in captured


def test_open_zarr_preserves_creation_keywords_for_zarr3(monkeypatch, tmp_path):
    captured = {}

    def fake_open(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(utils.zarr, "__version__", "3.2.1")
    monkeypatch.setattr(utils.zarr, "open", fake_open)

    config = {"write_empty_chunks": False}
    utils.open_zarr(
        str(tmp_path / "output.zarr"),
        mode="w",
        shape=(1,),
        dtype="u1",
        zarr_format=2,
        config=config,
    )

    assert captured["zarr_format"] == 2
    assert captured["config"] == config
    assert "zarr_version" not in captured
    assert "write_empty_chunks" not in captured
