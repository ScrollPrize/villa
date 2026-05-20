from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from vesuvius.scripts import extract_sparse_ome_zarr


@pytest.mark.parametrize(
    ("store_path", "expected_options"),
    [
        (
            "s3://vesuvius-challenge-open-data/PHercParis4/volumes/example.zarr/",
            {"anon": True},
        ),
        ("s3://private-bucket/example.zarr/", {"anon": False}),
    ],
)
def test_open_zarr_read_passes_s3_storage_options(monkeypatch, store_path, expected_options):
    calls = []

    fake_fsspec = SimpleNamespace(
        get_mapper=lambda path, **kwargs: calls.append((path, kwargs)) or "mapper"
    )
    monkeypatch.setitem(sys.modules, "fsspec", fake_fsspec)
    monkeypatch.setattr(extract_sparse_ome_zarr.zarr, "open", lambda store, mode: (store, mode))

    assert extract_sparse_ome_zarr.open_zarr_read(store_path) == ("mapper", "r")
    assert calls == [(store_path, expected_options)]
