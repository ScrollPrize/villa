"""Chunk caching for remote reads.

Without a cache every read goes to the network, so overlapping patches
re-download the same chunks. These tests pin down when the cache is applied,
when it must stay out of the way, and that reads through it return identical
data.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import zarr

from vesuvius.data.utils import _CACHE_ENV, _chunk_cache_url, open_zarr

S3_URL = "s3://bucket/vol.zarr/0"
HTTP_URL = "https://example.org/vol.zarr/0"


# --------------------------------------------------------------------------
# URL rewriting
# --------------------------------------------------------------------------

def test_no_cache_dir_leaves_url_untouched() -> None:
    path, opts = _chunk_cache_url(S3_URL, {"anon": True}, None)
    assert path == S3_URL
    assert opts == {"anon": True}


def test_empty_cache_dir_leaves_url_untouched() -> None:
    path, opts = _chunk_cache_url(S3_URL, {"anon": True}, "")
    assert path == S3_URL


@pytest.mark.parametrize("url,protocol", [(S3_URL, "s3"), (HTTP_URL, "https")])
def test_cache_dir_wraps_url_and_nests_options(url: str, protocol: str) -> None:
    original = {"anon": True, "config_kwargs": {"x": 1}}
    path, opts = _chunk_cache_url(url, original, "/tmp/cache")

    assert path == f"simplecache::{url}"
    # the backend's options move under its own protocol key
    assert opts[protocol] == original
    assert opts["simplecache"] == {"cache_storage": "/tmp/cache"}
    # the caller's dict must not be mutated
    assert original == {"anon": True, "config_kwargs": {"x": 1}}


# --------------------------------------------------------------------------
# When open_zarr applies it
# --------------------------------------------------------------------------

@pytest.fixture
def captured(monkeypatch):
    """Capture what open_zarr hands to zarr.open without touching the network."""
    seen = {}

    def fake_open(path, **kwargs):
        seen["path"] = path
        seen["storage_options"] = kwargs.get("storage_options")
        return "array"

    monkeypatch.setattr(zarr, "open", fake_open)

    # open_zarr creates the parent prefix for write modes; that call needs AWS
    # credentials and is unrelated to caching.
    import fsspec

    class _NoopFS:
        def makedirs(self, *args, **kwargs):
            return None

    monkeypatch.setattr(fsspec, "filesystem", lambda *a, **k: _NoopFS())
    return seen


def test_remote_read_uses_cache_when_requested(captured) -> None:
    open_zarr(S3_URL, mode="r", cache_dir="/tmp/c")
    assert captured["path"] == f"simplecache::{S3_URL}"
    assert captured["storage_options"]["simplecache"] == {"cache_storage": "/tmp/c"}


def test_remote_read_without_cache_dir_is_unchanged(captured, monkeypatch) -> None:
    monkeypatch.delenv(_CACHE_ENV, raising=False)
    open_zarr(S3_URL, mode="r")
    assert captured["path"] == S3_URL


def test_environment_variable_enables_cache(captured, monkeypatch) -> None:
    monkeypatch.setenv(_CACHE_ENV, "/tmp/from-env")
    open_zarr(S3_URL, mode="r")
    assert captured["path"] == f"simplecache::{S3_URL}"
    assert captured["storage_options"]["simplecache"]["cache_storage"] == "/tmp/from-env"


def test_explicit_argument_beats_environment(captured, monkeypatch) -> None:
    monkeypatch.setenv(_CACHE_ENV, "/tmp/from-env")
    open_zarr(S3_URL, mode="r", cache_dir="/tmp/explicit")
    assert captured["storage_options"]["simplecache"]["cache_storage"] == "/tmp/explicit"


def test_local_path_never_wrapped(captured, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(_CACHE_ENV, "/tmp/c")
    open_zarr(str(tmp_path / "local.zarr"), mode="r")
    assert "simplecache" not in captured["path"]
    # local stores must not receive storage_options at all
    assert captured["storage_options"] is None


@pytest.mark.parametrize("mode", ["w", "a", "r+"])
def test_write_modes_never_cached(captured, monkeypatch, mode: str) -> None:
    """Caching a store that is being written would serve stale chunks."""
    monkeypatch.setenv(_CACHE_ENV, "/tmp/c")
    open_zarr(S3_URL, mode=mode)
    assert captured["path"] == S3_URL


# --------------------------------------------------------------------------
# Data integrity through the cache
# --------------------------------------------------------------------------

@pytest.mark.network
def test_cached_read_matches_uncached_and_avoids_refetch(tmp_path: Path) -> None:
    """Same bytes, and the second read does not go to the network."""
    import s3fs

    url = ("s3://vesuvius-challenge-open-data/PHercParis4/volumes/"
           "20260411134726-2.400um-0.2m-78keV-masked.zarr/3")
    sel = (slice(4736, 4740), slice(2000, 2064), slice(2000, 2064))

    fetches = {"n": 0}
    original = s3fs.S3FileSystem._cat_file

    async def counting(self, *args, **kwargs):
        fetches["n"] += 1
        return await original(self, *args, **kwargs)

    s3fs.S3FileSystem._cat_file = counting
    try:
        plain = np.asarray(open_zarr(url, mode="r",
                                     storage_options={"anon": True})[sel])

        cache_dir = str(tmp_path / "chunks")
        arr = open_zarr(url, mode="r", storage_options={"anon": True},
                        cache_dir=cache_dir)
        first = np.asarray(arr[sel])
        after_first = fetches["n"]
        second = np.asarray(arr[sel])
    finally:
        s3fs.S3FileSystem._cat_file = original

    np.testing.assert_array_equal(plain, first)
    np.testing.assert_array_equal(first, second)
    assert fetches["n"] == after_first, "second read should be served from cache"
    assert any(Path(cache_dir).rglob("*")), "cache directory should be populated"
