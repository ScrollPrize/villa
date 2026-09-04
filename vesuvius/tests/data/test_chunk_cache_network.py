"""Network-marked check of the persistent disk chunk cache against real scroll
data. Skipped by default; run with:

    uv run pytest -m 'slow and network' \
        tests/data/test_chunk_cache_network.py -v -s

The local tests stub the remote store out, so nothing there proves the cache
behaves against a real HTTPS zarr. These read one 128³ chunk from a high
pyramid level of the Scroll 1A standardized volume, which keeps the whole file
to a couple of megabytes over the wire, and check three things: cached reads
are byte-identical to uncached ones, a second store on the same cache
directory fetches no chunks at all, and the cache directory really holds the
bytes.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

import numpy as np
import pytest

from vesuvius.data.utils import open_zarr


pytestmark = [pytest.mark.slow, pytest.mark.network]


VOLUME_URL = (
    "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/"
    "volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr"
)
# Level 5 of the pyramid is (450, 247, 253) in 128³ chunks, so the region
# below is exactly one chunk: one request, a few MB.
ARRAY_URL = f"{VOLUME_URL}/5"
REGION = (slice(128, 256), slice(0, 128), slice(0, 128))

_METADATA_NAMES = ("zarr.json", ".zarray", ".zgroup", ".zattrs", ".zmetadata")


def _is_metadata(url: str) -> bool:
    return url.rsplit("/", 1)[-1] in _METADATA_NAMES


@contextmanager
def _counted_http_requests():
    """Collect the URLs fsspec fetches over HTTP inside the block.

    Wrapping the filesystem itself counts what actually leaves the machine,
    rather than trusting the cache store's own bookkeeping.
    """
    # request counting after the method in issue #1325
    from fsspec.implementations.http import HTTPFileSystem

    original = HTTPFileSystem._cat_file
    urls: list[str] = []

    async def counting(self, url, *args, **kwargs):
        urls.append(str(url))
        return await original(self, url, *args, **kwargs)

    HTTPFileSystem._cat_file = counting
    try:
        yield urls
    finally:
        HTTPFileSystem._cat_file = original


def _read_region(array) -> np.ndarray:
    return np.asarray(array[REGION])


@pytest.fixture(scope="module")
def uncached_patch() -> np.ndarray:
    """The reference read, straight from the remote store."""
    with _counted_http_requests() as urls:
        patch = _read_region(open_zarr(ARRAY_URL, mode="r"))
    chunk_urls = [u for u in urls if not _is_metadata(u)]
    print(f"\nuncached read: {len(chunk_urls)} chunk request(s), shape {patch.shape}")
    assert chunk_urls, "expected the uncached read to fetch at least one chunk"
    return patch


@pytest.fixture(scope="module")
def cache_root(tmp_path_factory):
    return tmp_path_factory.mktemp("chunk_cache")


@pytest.fixture(scope="module")
def cold_cache_read(cache_root):
    """Populate a fresh cache directory, returning the patch and its requests."""
    with _counted_http_requests() as urls:
        started = time.perf_counter()
        patch = _read_region(open_zarr(ARRAY_URL, mode="r", chunk_cache_dir=str(cache_root)))
        elapsed = time.perf_counter() - started
    chunk_urls = [u for u in urls if not _is_metadata(u)]
    print(f"cold cache read: {len(chunk_urls)} chunk request(s) in {elapsed:.1f} s")
    return patch, chunk_urls


def test_cached_read_is_byte_identical(uncached_patch, cold_cache_read):
    patch, chunk_urls = cold_cache_read
    assert chunk_urls, "a cold cache should still have to fetch the chunk"
    assert patch.dtype == uncached_patch.dtype
    assert patch.shape == uncached_patch.shape
    assert np.array_equal(patch, uncached_patch)


def test_second_store_fetches_no_chunks(cache_root, cold_cache_read, uncached_patch):
    """A new store over a warm cache directory must not touch the network."""
    with _counted_http_requests() as urls:
        started = time.perf_counter()
        patch = _read_region(open_zarr(ARRAY_URL, mode="r", chunk_cache_dir=str(cache_root)))
        elapsed = time.perf_counter() - started
    chunk_urls = [u for u in urls if not _is_metadata(u)]
    print(
        f"warm cache read: {len(chunk_urls)} chunk request(s), "
        f"{len(urls)} request(s) total, in {elapsed:.2f} s"
    )
    assert chunk_urls == []
    assert np.array_equal(patch, uncached_patch)


def test_cache_directory_holds_the_chunk(cache_root, cold_cache_read):
    """Cached bytes land under a subtree namespaced by the source URL."""
    scheme, _, rest = ARRAY_URL.partition("://")
    namespaced = cache_root / scheme / rest
    assert namespaced.is_dir(), f"no cache subtree at {namespaced}"

    sizes = [
        os.path.getsize(os.path.join(dirpath, name))
        for dirpath, _dirnames, filenames in os.walk(namespaced)
        for name in filenames
    ]
    print(f"cache subtree {namespaced}: {len(sizes)} file(s), {sum(sizes) / 2**20:.2f} MB")
    assert sizes, "cache directory is empty"
    assert sum(sizes) > 0
