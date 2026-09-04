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
import zarr

from vesuvius.ink_detection import volume_io
from vesuvius.ink_detection.volume_io import (
    disk_cache_subdir,
    open_volume,
    open_volume_root,
)


_ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3


def _write_pyramid(path: Path, array: np.ndarray) -> None:
    root = zarr.open_group(path, mode="w")
    if _ZARR_V3:
        root.create_array("0", data=array, chunks=(3, 4, 5))
    else:
        root.create_dataset("0", data=array, chunks=(3, 4, 5))


def _shared_cache_worker(source_path, cache_dir, expected_digest, start, results):
    try:
        volume = open_volume(source_path, 0, cache_dir=cache_dir)
        start.wait()
        digests = [
            hashlib.sha256(np.asarray(volume[:]).tobytes()).hexdigest()
            for _ in range(6)
        ]
        results.put((True, digests, expected_digest))
    except Exception as exc:
        results.put((False, repr(exc), expected_digest))


def _forked_dataset_read(dataset, results) -> None:
    results.put((os.getpid(), dataset._open("volume.zarr", 0)))


@pytest.mark.skipif(not _ZARR_V3, reason="CacheStore requires Zarr 3")
def test_cache_store_hits_source_free_and_persists_full_values(tmp_path):
    from zarr.core.buffer.core import default_buffer_prototype
    from zarr.experimental.cache_store import CacheStore
    from zarr.storage import LocalStore

    class GuardedMemoryStore(zarr.storage.MemoryStore):
        def __init__(self) -> None:
            super().__init__(read_only=False)
            self.get_calls = []
            self.fail_reads = False

        async def get(self, key, prototype=None, byte_range=None):
            self.get_calls.append((key, byte_range))
            if self.fail_reads:
                raise AssertionError(f"source fetch for cached key {key!r}")
            return await super().get(key, prototype, byte_range)

    async def scenario():
        prototype = default_buffer_prototype()
        source = GuardedMemoryStore()
        await source.set("c/1/2/3", prototype.buffer.from_bytes(b"encoded-chunk"))
        cache_path = tmp_path / "cache"
        first = CacheStore(
            store=source,
            cache_store=LocalStore(cache_path),
            max_size=None,
        )

        first_value = await first.get("c/1/2/3", prototype)
        assert first_value.to_bytes() == b"encoded-chunk"
        assert source.get_calls == [("c/1/2/3", None)]

        source.get_calls.clear()
        source.fail_reads = True
        repeated = await first.get("c/1/2/3", prototype)
        assert repeated.to_bytes() == b"encoded-chunk"
        assert source.get_calls == []

        fresh = CacheStore(
            store=source,
            cache_store=LocalStore(cache_path),
            max_size=None,
        )
        persisted = await fresh.get("c/1/2/3", prototype)
        assert persisted.to_bytes() == b"encoded-chunk"
        assert source.get_calls == []

    asyncio.run(scenario())


@pytest.mark.skipif(not _ZARR_V3, reason="volume disk cache requires Zarr 3")
def test_cached_and_uncached_volume_reads_are_byte_identical(tmp_path):
    source_path = tmp_path / "volume.zarr"
    expected = (
        np.arange(7 * 11 * 13, dtype=np.uint16).reshape(7, 11, 13) * 17 + 3
    )
    _write_pyramid(source_path, expected)

    uncached = np.asarray(open_volume(source_path, 0)[:])
    cached = np.asarray(
        open_volume(source_path, 0, cache_dir=tmp_path / "cache")[:]
    )

    assert uncached.dtype == cached.dtype == expected.dtype
    assert uncached.shape == cached.shape == expected.shape
    assert uncached.tobytes() == cached.tobytes() == expected.tobytes()
    assert any(
        path.is_file()
        for path in disk_cache_subdir(str(source_path), tmp_path / "cache").rglob("*")
    )


@pytest.mark.skipif(not _ZARR_V3, reason="volume disk cache requires Zarr 3")
def test_open_sweep_prunes_oldest_recursively_and_leaves_under_budget(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(volume_io.zarr, "open", lambda *, store, mode: store)
    cache_root = tmp_path / "cache"

    over_path = disk_cache_subdir("over.zarr", cache_root)
    over_files = [
        over_path / "zarr.json",
        over_path / "c" / "0" / "0" / "0",
        over_path / "c" / "0" / "0" / "1",
    ]
    for timestamp, path in enumerate(over_files, start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([timestamp]) * 4)
        os.utime(path, ns=(timestamp, timestamp))

    opened = open_volume_root(
        "over.zarr",
        cache_dir=cache_root,
        cache_max_gb=10 / 1e9,
    )
    assert opened.max_size == 10
    assert not over_files[0].exists()
    assert over_files[1].read_bytes() == b"\x02" * 4
    assert over_files[2].read_bytes() == b"\x03" * 4
    assert sum(size for _, size, _ in volume_io._cache_snapshot(over_path)) == 8

    under_path = disk_cache_subdir("under.zarr", cache_root)
    under_files = [under_path / "zarr.json", under_path / "c" / "0"]
    for timestamp, (path, size) in enumerate(zip(under_files, (4, 6)), start=20):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        os.utime(path, ns=(timestamp, timestamp))
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in under_files]

    open_volume_root(
        "under.zarr",
        cache_dir=cache_root,
        cache_max_gb=10 / 1e9,
    )

    after = [(path.read_bytes(), path.stat().st_mtime_ns) for path in under_files]
    assert after == before


@pytest.mark.skipif(not _ZARR_V3, reason="volume disk cache requires Zarr 3")
def test_shared_cache_multiprocess_reads_are_not_torn(tmp_path):
    source_path = tmp_path / "source.zarr"
    cache_dir = tmp_path / "cache"
    expected = (
        np.arange(8 * 9 * 10, dtype=np.uint16).reshape(8, 9, 10) * 13 + 5
    )
    _write_pyramid(source_path, expected)
    expected_digest = hashlib.sha256(expected.tobytes()).hexdigest()
    context = mp.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_shared_cache_worker,
            args=(str(source_path), str(cache_dir), expected_digest, start, results),
        )
        for _ in range(4)
    ]
    for process in processes:
        process.start()
    start.set()
    observations = [results.get(timeout=60) for _ in processes]
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0
    assert all(success for success, _, _ in observations), observations
    assert all(
        all(digest == expected for digest in digests)
        for _, digests, expected in observations
    )


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


@pytest.mark.skipif(not _ZARR_V3, reason="direct FsspecStore is a Zarr-3 path")
@pytest.mark.parametrize("with_cache", [False, True])
def test_remote_open_keeps_authored_url_and_sets_process_options(
    tmp_path, monkeypatch, with_cache
):
    captured = {}

    class ArrayRoot:
        shape = (1, 1, 1)

    def fake_from_url(url, *, storage_options, read_only):
        captured["url"] = url
        captured["options"] = storage_options
        captured["read_only"] = read_only
        return zarr.storage.MemoryStore(read_only=True)

    monkeypatch.setattr(
        volume_io.zarr.storage.FsspecStore,
        "from_url",
        staticmethod(fake_from_url),
    )
    monkeypatch.setattr(
        volume_io.zarr, "open", lambda *, store, mode: ArrayRoot()
    )
    path = "s3://vesuvius-challenge-open-data/a.zarr/"
    cache_kwargs = {"cache_dir": tmp_path} if with_cache else {}

    result = open_volume(path, 0, **cache_kwargs)

    assert isinstance(result, ArrayRoot)
    assert captured == {
        "url": path,
        "options": {"anon": True, "skip_instance_cache": True},
        "read_only": True,
    }


@pytest.mark.skipif(not _ZARR_V3, reason="direct FsspecStore is a Zarr-3 path")
def test_public_substring_http_uses_real_backend_without_s3_options(tmp_path):
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            del format, args

    source_path = tmp_path / "vesuvius-challenge-open-data.zarr"
    expected = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    zarr.open_array(
        source_path,
        mode="w",
        shape=expected.shape,
        chunks=expected.shape,
        dtype=expected.dtype,
        zarr_format=2,
    )[:] = expected
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(QuietHandler, directory=str(tmp_path)),
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    opened = None
    try:
        url = (
            f"http://127.0.0.1:{server.server_port}/"
            "vesuvius-challenge-open-data.zarr"
        )
        opened = open_volume_root(url)
        np.testing.assert_array_equal(opened[:], expected)
    finally:
        if opened is not None:
            filesystem = opened.store.fs
            opened.store.close()
            filesystem.close_session(filesystem.loop, filesystem._session)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=10)


@pytest.mark.skipif(not _ZARR_V3, reason="direct FsspecStore is a Zarr-3 path")
def test_private_https_basic_auth_is_unchanged(tmp_path, monkeypatch):
    captured = {}

    def fake_from_url(url, *, storage_options, read_only):
        captured.update(
            url=url, storage_options=storage_options, read_only=read_only
        )
        return zarr.storage.MemoryStore(read_only=True)

    monkeypatch.setattr(
        volume_io.zarr.storage.FsspecStore,
        "from_url",
        staticmethod(fake_from_url),
    )
    monkeypatch.setattr(volume_io.zarr, "open", lambda *, store, mode: store)
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        '{"username": "reader", "password": "secret"}', encoding="utf-8"
    )

    with pytest.warns(DeprecationWarning, match="BasicAuth"):
        open_volume_root("https://example.test/volume.zarr", auth_path)

    assert captured["url"] == "https://example.test/volume.zarr"
    assert captured["read_only"] is True
    assert captured["storage_options"]["skip_instance_cache"] is True
    assert tuple(captured["storage_options"]["client_kwargs"]["auth"])[:2] == (
        "reader",
        "secret",
    )


def test_zarr2_cached_open_raises_factually(tmp_path):
    if _ZARR_V3:
        return
    with pytest.raises(NotImplementedError, match="volume disk cache requires zarr 3"):
        open_volume_root(tmp_path / "volume.zarr", cache_dir=tmp_path / "cache")


def test_public_group_keys_include_arrays_and_groups(tmp_path):
    path = tmp_path / "mixed.zarr"
    kwargs = {"mode": "w"}
    if _ZARR_V3:
        kwargs["zarr_format"] = 2
    root = zarr.open_group(path, **kwargs)
    root.create_group("child_group")
    if _ZARR_V3:
        root.create_array("child_array", shape=(1,), chunks=(1,), dtype="u1")
    else:
        root.create_dataset("child_array", shape=(1,), chunks=(1,), dtype="u1")
    reopened = zarr.open_group(path, mode="r")

    assert sorted(str(key) for key in reopened.keys()) == [
        "child_array",
        "child_group",
    ]
    assert volume_io._available_top_level_keys(reopened) == (
        "child_array",
        "child_group",
    )


def test_missing_resolution_uses_node_error_when_listing_fails(monkeypatch):
    class FailingGroup:
        def __getitem__(self, key):
            raise KeyError(key)

        def keys(self):
            raise RuntimeError("listing unavailable")

    monkeypatch.setattr(
        volume_io, "open_vesuvius_zarr", lambda *args, **kwargs: FailingGroup()
    )
    error_type = getattr(zarr.errors, "NodeNotFoundError", None)
    if error_type is None:
        error_type = getattr(zarr.errors, "PathNotFoundError", KeyError)
    with pytest.raises(error_type, match="resolution '3'") as raised:
        open_volume("missing.zarr", 3)
    assert isinstance(raised.value.__cause__, KeyError)

    class ArrayRoot:
        shape = (2, 2, 2)

    monkeypatch.setattr(
        volume_io, "open_vesuvius_zarr", lambda *args, **kwargs: ArrayRoot()
    )
    with pytest.raises(error_type, match="zarr array"):
        open_volume("array.zarr", 2)
    assert open_volume(
        "array.zarr", 2, root_array_is_requested_level=True
    ).shape == (2, 2, 2)


def test_missing_node_error_has_zarr2_fallback(monkeypatch):
    class CompatiblePathError(Exception):
        pass

    monkeypatch.delattr(volume_io.zarr.errors, "NodeNotFoundError", raising=False)
    monkeypatch.setattr(
        volume_io.zarr.errors,
        "PathNotFoundError",
        CompatiblePathError,
        raising=False,
    )
    assert isinstance(
        volume_io._missing_node_error("missing"), CompatiblePathError
    )


RECIPE_TAG = "level2-zmean4-21slice-v1"


def _prepared_root(tmp_path, attrs, name="input.zarr"):
    """A prepared input is a group carrying the stamp `prepare_9um` writes."""

    root = zarr.open_group(str(tmp_path / name), mode="w")
    root.attrs.update(attrs)
    return root


def _ome_attrs(scales):
    """Minimal OME metadata of a rendered surface volume: micrometre axes per level."""

    return {
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": str(level),
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [scales[0], um, um]},
                            {"type": "translation", "translation": [0.0, 0.0, 0.0]},
                        ],
                    }
                    for level, um in enumerate(scales)
                ],
            }
        ]
    }


def test_input_at_the_scale_the_recipe_trains_at_is_silent(tmp_path):
    root = _prepared_root(
        tmp_path,
        {"format": RECIPE_TAG, "source_level": "2", "source": "render.zarr"},
    )
    reader = lambda path: _ome_attrs([2.399, 4.798, 9.596])
    assert (
        volume_io.check_flat_input_scale(
            root, source="input.zarr", attrs_reader=reader
        )
        is None
    )


def test_input_prepared_at_the_wrong_level_reports_the_numbers(tmp_path):
    root = _prepared_root(
        tmp_path,
        {"format": RECIPE_TAG, "source_level": "0", "source": "render.zarr"},
    )
    reader = lambda path: _ome_attrs([2.399, 4.798, 9.596])
    message = volume_io.check_flat_input_scale(
        root, source="input.zarr", attrs_reader=reader
    )
    assert message is not None
    assert "2.399 um in-plane" in message
    assert "9.596 or 9.362 um" in message
    assert "nearest 9.362" in message  # nearest in relative distance
    assert "factor 0.256" in message


def test_strict_refuses_what_the_default_reports(tmp_path):
    root = _prepared_root(
        tmp_path,
        {"format": RECIPE_TAG, "source_level": "0", "source": "render.zarr"},
    )
    reader = lambda path: _ome_attrs([2.399, 4.798, 9.596])
    with pytest.raises(ValueError, match="factor 0.25"):
        volume_io.check_flat_input_scale(
            root, source="input.zarr", strict=True, attrs_reader=reader
        )


def test_unreachable_source_falls_back_and_never_raises(tmp_path):
    root = _prepared_root(
        tmp_path,
        {"format": RECIPE_TAG, "source_level": "0", "source": "gone.zarr"},
    )

    def reader(path):
        raise FileNotFoundError(path)

    message = volume_io.check_flat_input_scale(
        root, source="input.zarr", attrs_reader=reader
    )
    assert message is not None
    assert "absolute scale not checked" in message
    assert "source unreachable" in message


def test_unreachable_source_at_the_expected_level_is_silent(tmp_path):
    # Nothing is known about the scale and nothing is inconsistent: say nothing.
    root = _prepared_root(
        tmp_path,
        {"format": RECIPE_TAG, "source_level": "2", "source": "gone.zarr"},
    )

    def reader(path):
        raise FileNotFoundError(path)

    assert (
        volume_io.check_flat_input_scale(
            root, source="input.zarr", attrs_reader=reader
        )
        is None
    )


def test_input_with_no_attrs_at_all_is_left_alone(tmp_path):
    # What anyone preparing inputs by other means will hand this code.
    root = _prepared_root(tmp_path, {})
    assert volume_io.check_flat_input_scale(root, source="input.zarr") is None


def test_bare_array_without_attrs_is_left_alone(tmp_path):
    array = zarr.open_array(
        str(tmp_path / "bare.zarr"), mode="w", shape=(21, 8, 8), dtype="uint8"
    )
    assert volume_io.check_flat_input_scale(array, source="bare.zarr") is None


def test_input_without_known_format_tag_is_left_alone(tmp_path):
    root = _prepared_root(tmp_path, _ome_attrs([2.399, 4.798, 9.596]))
    assert volume_io.check_flat_input_scale(root, source="input.zarr") is None


def test_missing_source_level_is_reported(tmp_path):
    root = _prepared_root(tmp_path, {"format": RECIPE_TAG, "source": "render.zarr"})
    message = volume_io.check_flat_input_scale(root, source="input.zarr")
    assert message is not None and "records no source_level" in message


def test_source_recording_only_a_voxel_size_is_scaled_by_level(tmp_path):
    # A store that records its scale outside OME metadata, as Volume::normalize reads it.
    root = _prepared_root(
        tmp_path,
        {"format": RECIPE_TAG, "source_level": "2", "source": "volume.zarr"},
    )
    reader = lambda path: {"voxelsize": 2.399}
    assert (
        volume_io.check_flat_input_scale(
            root, source="input.zarr", attrs_reader=reader
        )
        is None
    )


def test_sample_pixel_size_is_read_in_millimetres(tmp_path):
    doc = {"scan": {"tomo": {"acquisition": {"detector": {"samplePixelSize": 0.002399}}}}}
    assert volume_io.voxel_size_um_from_document(doc) == pytest.approx(2.399)


def test_native_nine_micron_render_at_level_zero_is_silent(tmp_path):
    """The recipe's second documented family: a 9.362 um render used at level 0.

    prepare_9um stamps the same tag whatever `--level` it was run with, so this input
    carries a level-2 tag with source_level 0 and is still correct.
    """

    root = _prepared_root(
        tmp_path,
        {"format": RECIPE_TAG, "source_level": "0", "source": "native9.zarr"},
    )
    reader = lambda path: _ome_attrs([9.362])
    assert (
        volume_io.check_flat_input_scale(
            root, source="input.zarr", attrs_reader=reader
        )
        is None
    )


def test_native_nine_micron_recorded_as_voxel_size_is_silent(tmp_path):
    root = _prepared_root(
        tmp_path,
        {"format": RECIPE_TAG, "source_level": "0", "source": "native9.zarr"},
    )
    reader = lambda path: {"voxelsize": 9.362}
    assert (
        volume_io.check_flat_input_scale(
            root, source="input.zarr", attrs_reader=reader
        )
        is None
    )


def test_refusal_uses_a_dedicated_exception(tmp_path):
    root = _prepared_root(
        tmp_path,
        {"format": RECIPE_TAG, "source_level": "0", "source": "render.zarr"},
    )
    reader = lambda path: _ome_attrs([2.399, 4.798, 9.596])
    with pytest.raises(volume_io.InputScaleRefused):
        volume_io.check_flat_input_scale(
            root, source="input.zarr", strict=True, attrs_reader=reader
        )
