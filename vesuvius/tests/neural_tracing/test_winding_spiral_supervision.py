"""CPU tests for ``infer_winding_volume.py --spiral-supervision-only``.

The real ``gpu_worker`` and ``main`` run in-process with the volume read and
the neural forward replaced by fixed synthetic phase fields, so the direct
store can be compared with the cache-then-export store for the same fields.
"""

import importlib
import importlib.util
import json
import queue
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import zarr

from vesuvius.neural_tracing.winding_models import (
    export_spiral_supervision as exporter,
    infer_winding_volume as iwv,
)

# The native phase cache is written with the zarr v3 API; the direct store
# is plain .npy files.
requires_zarr_v3 = pytest.mark.skipif(
    int(zarr.__version__.split(".", 1)[0]) < 3,
    reason="the native phase cache needs zarr>=3")

TRANSVERSE = 8
RAY_LENGTH = 64
NUM_RAYS = 70
PART_RAYS = 16
SKIPPED = {5, 33, 34}  # slabs whose volume chunks are "truncated"
MALFORMED = "decoded chunk byte size does not match full chunk shape"
ARRAYS = ("ray_origin_zyx", "ray_step_zyx", "crossing_offsets",
          "crossing_t", "crossing_level", "seed_winding")


def _fields(seed=0):
    """Monotone headless-model phase and validity for every slab."""
    rng = np.random.default_rng(seed)
    shape = (NUM_RAYS, TRANSVERSE, TRANSVERSE, RAY_LENGTH)
    increments = rng.uniform(0.02, 0.3, size=shape).astype(np.float32)
    phase = np.cumsum(increments, axis=-1, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)
    valid[3, :, :, 40:44] = False  # hole after the anchor: run stops there
    valid[7, :, :, 28:36] = False  # invalid anchor: slab is not retained
    valid[11, :, :, :20] = False
    return phase, valid


class _InlineProcess:
    def __init__(self, target, args):
        self.target, self.args, self.exitcode = target, args, None

    def start(self):
        self.target(*self.args)
        self.exitcode = 0

    def is_alive(self):
        return False

    def join(self):
        pass


class _InlineContext:
    Queue = queue.Queue

    def Process(self, target, args):
        return _InlineProcess(target, args)


class _NoModel:
    def __init__(self, cfg):
        pass

    def load_state_dict(self, state):
        pass

    def to(self, device):
        return self

    def eval(self):
        return self


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Return ``run(output, *mode_args, fail_at=None, gpus="0,1")``, which
    runs main() and returns the extracted ray ids."""
    monkeypatch.setattr(iwv, "SUPERVISION_PART_RAYS", PART_RAYS)
    if importlib.util.find_spec("vc") is None:
        # volume_slab_extractor imports the native bindings at module level;
        # nothing here samples a volume.
        monkeypatch.setitem(sys.modules, "vc", types.ModuleType("vc"))
    extractor = importlib.import_module(
        "vesuvius.neural_tracing.winding_models.volume_slab_extractor")
    winding_model = importlib.import_module(
        "vesuvius.neural_tracing.winding_models.winding_model")
    monkeypatch.setattr(winding_model, "WindingModel", _NoModel)

    rng = np.random.default_rng(1)
    directions = rng.normal(size=(NUM_RAYS, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    seeds_path = tmp_path / "seed_rays.npz"
    np.savez(
        seeds_path,
        seed_xyz=np.stack([1000.0 + 7.0 * np.arange(NUM_RAYS),
                           2000.0 + 3.0 * np.arange(NUM_RAYS),
                           np.full(NUM_RAYS, 3000.0)], axis=1).astype(np.float32),
        direction_xyz=directions.astype(np.float32),
        seed_winding=(10 + np.arange(NUM_RAYS) % 5).astype(np.int16),
        z_range=np.array([2900, 3100]),
        seed_windings=np.arange(10, 15),
        dr_per_winding=16.0,
        umbilicus_path="umbilicus.json",
    )
    model_path = tmp_path / "model.pth"
    torch.save({
        "config": {
            "model": {"use_crossing_head": False},
            "ray_length": RAY_LENGTH, "transverse_size": TRANSVERSE,
            "column_stride": 1, "spacing": 1.0,
        },
        "model": {},
    }, model_path)
    phase, valid = _fields()
    midpoint = (RAY_LENGTH - 1) / 2.0

    def ray_id(direction, origin):
        seed = np.asarray(origin) + midpoint * np.asarray(direction)
        return int(round((seed[0] - 1000.0) / 7.0))

    def forward(batch, *args, **kwargs):
        ids = [int(item[1].flat[0]) for item in batch]
        return None, phase[ids], None

    def go(output, *mode_args, fail_at=None, gpus="0,1"):
        extracted = []

        def extract(self, volume_idx, direction, origin):
            index = ray_id(direction, origin)
            if index == fail_at:
                raise ValueError("simulated interruption")
            extracted.append(index)
            if index in SKIPPED:
                raise RuntimeError(MALFORMED)
            image = np.full(valid.shape[1:], index, dtype=np.float32)
            return image, valid[index], self.slab_frame(direction, origin)

        argv = [
            "infer_winding_volume.py", str(tmp_path / "fit.ckpt"), str(output),
            "--reference-zarr", str(tmp_path / "volume.zarr"),
            "--model-ckpt", str(model_path),
            "--seed-rays-npz", str(seeds_path),
            "--gpus", gpus, "--batch-size", "4", "--extract-threads", "1",
            *mode_args,
        ]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("multiprocessing.get_context",
                           lambda method: _InlineContext()), \
                mock.patch.object(extractor.VolumeSlabExtractor, "extract",
                                  extract), \
                mock.patch.object(iwv, "_forward_batch", forward):
            iwv.main()
        return extracted

    return go


def _store_arrays(root):
    """Concatenate a compact store's shards as WindingInferenceStore does."""
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    values = {key: [] for key in ARRAYS}
    base = 0
    for shard in manifest["shards"]:
        for key in ARRAYS:
            value = np.load(Path(root) / shard["name"]
                            / shard["arrays"][key]["file"])
            if key == "crossing_offsets":
                value = value[1:] + base if values[key] else value + base
                base = int(value[-1])
            values[key].append(value)
    return manifest, {key: np.concatenate(value) for key, value in values.items()}


@requires_zarr_v3
def test_direct_store_matches_cache_then_export(run, tmp_path):
    run(tmp_path / "cache.zarr", "--native-phase-only")
    exporter.export_spiral_supervision(
        tmp_path / "cache.zarr", tmp_path / "exported", workers=1,
        show_progress=False)
    run(tmp_path / "direct", "--spiral-supervision-only")

    cached, expected = _store_arrays(tmp_path / "exported")
    direct, actual = _store_arrays(tmp_path / "direct")
    for key in ARRAYS:
        assert actual[key].dtype == expected[key].dtype, key
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    # Slab 7 has an invalid anchor and keeps no crossings.
    assert direct["num_rays"] == cached["num_rays"]
    assert 0 < direct["num_rays"] < NUM_RAYS - len(SKIPPED)
    assert direct["num_crossings"] == cached["num_crossings"] > 0
    assert direct["export_mode"] == "direct"
    assert direct["source_cache"] is None
    attributes = direct["source_attributes"]
    assert attributes["num_slabs"] == NUM_RAYS
    assert attributes["num_available_slabs"] == NUM_RAYS - len(SKIPPED)
    for key in ("ray_length", "transverse_size", "column_stride", "spacing",
                "model_ckpt", "reference_zarr", "z_range", "seed_windings"):
        assert attributes[key] == cached["source_attributes"][key], key
    for key in ("center_native_column", "center_transverse_sample",
                "anchor_sample", "edge_margin", "edge_trim"):
        assert direct[key] == cached[key], key
    assert exporter.validate_spiral_supervision(tmp_path / "direct")[
        "num_crossings"] == direct["num_crossings"]
    assert not (tmp_path / "direct.partial").exists()


def test_interrupted_direct_run_resumes_from_finished_parts(run, tmp_path):
    run(tmp_path / "reference", "--spiral-supervision-only")
    with pytest.raises(ValueError, match="simulated interruption"):
        run(tmp_path / "direct", "--spiral-supervision-only", fail_at=40)
    assert not (tmp_path / "direct").exists()

    # Resume on one worker instead of two: parts are global ray ranges.
    extracted = run(tmp_path / "direct", "--spiral-supervision-only", gpus="0")

    # Parts 0-15 and 16-31 were written before slab 40 failed.
    assert sorted(extracted) == list(range(2 * PART_RAYS, NUM_RAYS))
    _, expected = _store_arrays(tmp_path / "reference")
    _, actual = _store_arrays(tmp_path / "direct")
    for key in ARRAYS:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def test_direct_store_refuses_existing_output(run, tmp_path):
    (tmp_path / "direct").mkdir()
    with pytest.raises(FileExistsError):
        run(tmp_path / "direct", "--spiral-supervision-only")


def test_direct_store_loads_in_spiral_fitting(run, tmp_path, monkeypatch):
    try:
        spiral_fitting = iwv._spiral_scripts_dir()
    except FileNotFoundError:
        pytest.skip("spiral-fitting checkout not found")
    monkeypatch.syspath_prepend(spiral_fitting)
    winding_supervision = pytest.importorskip("winding_supervision")
    run(tmp_path / "direct", "--spiral-supervision-only")

    store = winding_supervision.WindingInferenceStore(tmp_path / "direct", "cpu")
    manifest, arrays = _store_arrays(tmp_path / "direct")
    assert store.fingerprint["fingerprint"] == manifest["fingerprint"]
    assert len(store.origin) == manifest["num_rays"]
    np.testing.assert_array_equal(
        store.crossing_t.numpy(), arrays["crossing_t"])


@pytest.mark.parametrize("extra", [
    ["--column-step", "2"],
    ["--prob-volume"],
    ["--output-downsample=2"],
    ["--decode-workers", "2"],
    ["--native-phase-only"],
    ["--phase-cache", "cache.zarr"],
])
def test_direct_mode_rejects_reconstruction_options(extra):
    argv = ["infer_winding_volume.py", "fit.ckpt", "out", "--reference-zarr",
            "volume.zarr", "--model-ckpt", "model.pth",
            "--spiral-supervision-only", *extra]
    with mock.patch.object(sys, "argv", argv), pytest.raises(SystemExit):
        iwv.parse_args()
