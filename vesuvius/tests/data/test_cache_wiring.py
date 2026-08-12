"""The chunk cache settings have to survive the trip from the CLI down to open_zarr.

open_zarr grew cache_dir/cache_max_gb, but a setting nobody can reach from predict
is a setting nobody uses. These tests pin the forwarding at each seam: predict's
parser, VCDataset to Volume, and Volume to open_zarr.

Volume opens a store in three places and only two of them forwarded the cache
arguments; load_ome_metadata forwarded none, so even the in-memory cache never
applied to metadata reads. Every site is checked here so the next one added is
noticed.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from vesuvius.data.volume import Volume

SHAPE = (32, 32, 32)
CHUNKS = (16, 16, 16)


@pytest.fixture
def zarr_path(tmp_path) -> str:
    """A local store with no attributes, so metadata loading has to open it."""
    path = tmp_path / "vol.zarr"
    arr = zarr.open(str(path), mode="w", shape=SHAPE, chunks=CHUNKS, dtype="u1")
    arr[:] = np.zeros(SHAPE, dtype="u1")
    return str(path)


@pytest.fixture
def cache_dir(tmp_path) -> str:
    return str(tmp_path / "chunk-cache")


def _spy_open_zarr(monkeypatch, zarr_path: str) -> list[dict]:
    """Record the kwargs of every open_zarr call made from volume.py.

    The stand-in opens the local store plainly rather than delegating to the real
    open_zarr, so no cache store is ever constructed. That keeps these forwarding
    assertions identical across zarr majors: under zarr 2 the real open_zarr
    rejects cache=True outright (the in-memory CacheStore is a zarr 3 feature),
    which would turn a test about argument passing into a test about zarr's
    version.
    """
    from vesuvius.data import volume as volume_module

    calls: list[dict] = []

    def spy(**kwargs):
        calls.append(dict(kwargs))
        return zarr.open(zarr_path, mode="r")

    monkeypatch.setattr(volume_module, "open_zarr", spy)
    return calls


def _assert_forwarded(kwargs: dict, cache_dir: str) -> None:
    assert kwargs["cache"] is True
    assert kwargs["cache_size_mb"] == 8
    assert kwargs["cache_dir"] == cache_dir
    assert kwargs["cache_max_gb"] == 1.5


def _volume(zarr_path: str, cache_dir: str) -> Volume:
    return Volume(
        type="zarr",
        path=zarr_path,
        cache=True,
        cache_size_mb=8,
        cache_dir=cache_dir,
        cache_max_gb=1.5,
    )


def test_volume_init_forwards_cache_settings(zarr_path, cache_dir, monkeypatch):
    calls = _spy_open_zarr(monkeypatch, zarr_path)
    _volume(zarr_path, cache_dir)
    assert calls, "constructing a Volume opened no zarr store"
    for kwargs in calls:
        _assert_forwarded(kwargs, cache_dir)


def test_load_ome_metadata_forwards_cache_settings(zarr_path, cache_dir, monkeypatch):
    """The gap this task closes: metadata reads used to get no cache arguments."""
    calls = _spy_open_zarr(monkeypatch, zarr_path)
    vol = _volume(zarr_path, cache_dir)
    calls.clear()
    vol.load_ome_metadata()
    assert calls, "load_ome_metadata opened no zarr store"
    for kwargs in calls:
        _assert_forwarded(kwargs, cache_dir)


def test_load_data_forwards_cache_settings(zarr_path, cache_dir, monkeypatch):
    calls = _spy_open_zarr(monkeypatch, zarr_path)
    vol = _volume(zarr_path, cache_dir)
    calls.clear()
    vol.load_data()
    assert calls, "load_data opened no zarr store"
    for kwargs in calls:
        _assert_forwarded(kwargs, cache_dir)


def test_volume_defaults_leave_the_disk_cache_off(zarr_path, monkeypatch):
    """Defaults must reach open_zarr unchanged, or every existing caller shifts."""
    calls = _spy_open_zarr(monkeypatch, zarr_path)
    Volume(type="zarr", path=zarr_path)
    assert calls
    for kwargs in calls:
        assert kwargs["cache"] is False
        assert kwargs["cache_dir"] is None
        assert kwargs["cache_max_gb"] is None


def test_vc_dataset_forwards_cache_settings_to_volume(zarr_path, cache_dir, monkeypatch):
    from vesuvius.data import vc_dataset as vc_dataset_module

    # A real Volume, so the assertion covers the whole VCDataset -> Volume ->
    # open_zarr chain rather than just the first hop.
    calls = _spy_open_zarr(monkeypatch, zarr_path)
    seen: dict = {}
    real = vc_dataset_module.Volume

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(vc_dataset_module, "Volume", spy)
    vc_dataset_module.VCDataset(
        input_path=zarr_path,
        patch_size=(16, 16, 16),
        mode="infer",
        skip_empty_patches=False,
        verbose=False,
        cache=True,
        cache_dir=cache_dir,
        cache_max_gb=1.5,
    )
    assert seen, "VCDataset constructed no Volume"
    assert seen["cache"] is True
    assert seen["cache_dir"] == cache_dir
    assert seen["cache_max_gb"] == 1.5
    assert calls, "the Volume VCDataset built opened no zarr store"
    for kwargs in calls:
        assert kwargs["cache"] is True
        assert kwargs["cache_dir"] == cache_dir
        assert kwargs["cache_max_gb"] == 1.5
        # VCDataset exposes no cache_size_mb, so Volume's default has to survive.
        assert kwargs["cache_size_mb"] == 256


def test_vc_dataset_defaults_leave_the_cache_off(zarr_path, monkeypatch):
    from vesuvius.data import vc_dataset as vc_dataset_module

    seen: dict = {}
    real = vc_dataset_module.Volume

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(vc_dataset_module, "Volume", spy)
    vc_dataset_module.VCDataset(
        input_path=zarr_path,
        patch_size=(16, 16, 16),
        mode="infer",
        skip_empty_patches=False,
        verbose=False,
    )
    assert seen["cache"] is False
    assert seen["cache_dir"] is None
    assert seen["cache_max_gb"] is None


def test_predict_cli_accepts_the_cache_flags():
    # Importing predict pulls in the nnU-Net stack, which the volume-only install
    # does not have; the flags are still worth pinning where it is present.
    pytest.importorskip("nnunetv2", reason="predict imports the nnU-Net stack")
    from vesuvius.models.run.inference import build_parser

    args, _ = build_parser().parse_known_args(
        [
            "--model_path", "m",
            "--input_dir", "i",
            "--output_dir", "o",
            "--cache_dir", "/x",
            "--cache_max_gb", "2",
        ]
    )
    assert args.cache_dir == "/x"
    assert args.cache_max_gb == 2.0


def test_predict_cli_cache_flags_default_to_none():
    pytest.importorskip("nnunetv2", reason="predict imports the nnU-Net stack")
    from vesuvius.models.run.inference import build_parser

    args, _ = build_parser().parse_known_args(
        ["--model_path", "m", "--input_dir", "i", "--output_dir", "o"]
    )
    assert args.cache_dir is None
    assert args.cache_max_gb is None
