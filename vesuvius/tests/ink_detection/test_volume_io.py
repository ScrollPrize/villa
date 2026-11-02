from __future__ import annotations

import asyncio
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import multiprocessing as mp
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from vc_reader.reader import ZarrWriter

from vesuvius.ink_detection import volume_io
from vesuvius.ink_detection.volume_io import (
    open_volume,
    open_volume_root,
)


def _forked_dataset_read(dataset, results) -> None:
    results.put((os.getpid(), dataset._open("volume.zarr", 0)))


@pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(),
    reason="PID-key fork behavior requires multiprocessing fork",
)
def test_parent_populated_dataset_cache_misses_after_fork(monkeypatch):
    from vesuvius.ink_detection.data import dataset as dataset_module

    dataset = dataset_module.InkDataset.__new__(dataset_module.InkDataset)
    dataset.config = SimpleNamespace(
        volume_auth_json=None,
        volume_cache_dir=None,
        volume_cache_max_gb=None,
    )
    dataset._zarr_cache = {}

    def process_owned_open(*args, **kwargs):
        del args, kwargs
        return os.getpid()

    monkeypatch.setattr(dataset_module, "open_volume", process_owned_open)
    parent_pid = os.getpid()
    assert dataset._open("volume.zarr", 0) == parent_pid

    context = mp.get_context("fork")
    results = context.Queue()
    worker = context.Process(target=_forked_dataset_read, args=(dataset, results))
    with pytest.warns(DeprecationWarning, match="multi-threaded"):
        worker.start()
    worker_pid, opened_pid = results.get(timeout=30)
    worker.join(timeout=30)
    assert worker.exitcode == 0
    assert worker_pid != parent_pid
    assert opened_pid == worker_pid


def test_public_substring_http_uses_real_backend_without_s3_options(tmp_path):
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            del format, args

    source_path = tmp_path / "vesuvius-challenge-open-data.zarr"
    expected = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    ZarrWriter.create(
        source_path,
        shape=expected.shape,
        chunks=expected.shape,
        dtype=expected.dtype,
    )[:] = expected
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(QuietHandler, directory=str(tmp_path)),
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        url = (
            f"http://127.0.0.1:{server.server_port}/"
            "vesuvius-challenge-open-data.zarr"
        )
        opened = open_volume_root(url)
        np.testing.assert_array_equal(opened[:], expected)
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=10)


def test_missing_resolution_uses_node_error_when_listing_fails(monkeypatch):
    class FailingGroup:
        def __getitem__(self, key):
            raise KeyError(key)

        def keys(self):
            raise RuntimeError("listing unavailable")

    monkeypatch.setattr(
        volume_io, "open_node", lambda *args, **kwargs: FailingGroup()
    )
    error_type = FileNotFoundError
    with pytest.raises(error_type, match="resolution '3'") as raised:
        open_volume("missing.zarr", 3)
    assert isinstance(raised.value.__cause__, KeyError)

    class ArrayRoot:
        shape = (2, 2, 2)

    monkeypatch.setattr(
        volume_io, "open_node", lambda *args, **kwargs: ArrayRoot()
    )
    with pytest.raises(error_type, match="zarr array"):
        open_volume("array.zarr", 2)
    assert open_volume(
        "array.zarr", 2, root_array_is_requested_level=True
    ).shape == (2, 2, 2)


