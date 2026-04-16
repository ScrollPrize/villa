import pytest

from koine_machines.common import common


class StubStore:
    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs


def test_open_zarr_uses_anonymous_s3_for_public_vesuvius_bucket(monkeypatch):
    filesystem_calls = []
    zarr_open_calls = []

    def fake_filesystem(protocol, **kwargs):
        filesystem_calls.append((protocol, kwargs))
        return object()

    def fake_open(store, **kwargs):
        zarr_open_calls.append((store, kwargs))
        return object()

    monkeypatch.setattr(common.fsspec, "filesystem", fake_filesystem)
    monkeypatch.setattr(common.zarr.storage, "FSStore", StubStore)
    monkeypatch.setattr(common.zarr, "open", fake_open)

    common.open_zarr(
        "s3://vesuvius-challenge-open-data/example-volume.zarr/",
        resolution=0,
    )

    assert filesystem_calls == [("s3", {"anon": True})]
    assert zarr_open_calls[0][0].path == "s3://vesuvius-challenge-open-data/example-volume.zarr"
    assert zarr_open_calls[0][1] == {"path": "0", "mode": "r"}


def test_open_zarr_does_not_force_anonymous_s3_for_other_buckets(monkeypatch):
    filesystem_calls = []

    def fake_filesystem(protocol, **kwargs):
        filesystem_calls.append((protocol, kwargs))
        return object()

    monkeypatch.setattr(common.fsspec, "filesystem", fake_filesystem)
    monkeypatch.setattr(common.zarr.storage, "FSStore", StubStore)
    monkeypatch.setattr(common.zarr, "open", lambda *args, **kwargs: object())

    common.open_zarr("s3://private-bucket/example-volume.zarr/", resolution=0)

    assert filesystem_calls == [("s3", {})]


def test_open_zarr_path_not_found_names_store_resolution_and_available_keys(monkeypatch):
    class StubRoot:
        def group_keys(self):
            return ["0", "1"]

        def array_keys(self):
            return []

    def fake_open(store, **kwargs):
        if kwargs.get("path") == "4":
            raise common.zarr.errors.PathNotFoundError("4")
        return StubRoot()

    monkeypatch.setattr(common.zarr, "open", fake_open)

    with pytest.raises(common.zarr.errors.PathNotFoundError) as exc_info:
        common.open_zarr("/data/example-volume.zarr", resolution=4)

    message = str(exc_info.value)
    assert "/data/example-volume.zarr/4" in message
    assert "resolution '4'" in message
    assert "available top-level keys: 0, 1" in message
