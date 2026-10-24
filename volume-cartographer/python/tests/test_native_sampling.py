"""Binding regressions; select a CMake build using --vc-build-python."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import vc.volume as native

from test_sftp_volume import create_volume, expected


def test_bounded_cache_initialization():
    # Pool capacities are process-wide: exercise initialization in isolation.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(native.__file__).parents[1])
    subprocess.run([sys.executable, "-c", """
import sys
from importlib.machinery import PathFinder
# Prefer the selected on-disk build over the editable-install import redirect.
sys.meta_path.insert(0, PathFinder)
from vc.volume import initialize_chunk_cache, set_chunk_cache_decode_threads
initialize_chunk_cache(1048576, 2, 2, 2)
initialize_chunk_cache(2097152, 2, 2, 2)
set_chunk_cache_decode_threads(1)
for args in [(0,2,2,2), (1048576,2,0,2), (1048576,2,2,3)]:
    try:
        initialize_chunk_cache(*args)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError(args)
"""], check=True, env=env, timeout=30)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_exact_bbox_and_sampling(tmp_path, dtype):
    root = tmp_path / "volume.zarr"
    create_volume(root, dtype)
    (root / "metadata.json").write_text(json.dumps({
        "type": "vol", "uuid": "test-native-sampling", "name": "test",
        "width": 4, "height": 4, "slices": 4, "voxelsize": 1.0,
        "scan": {"voxelsize": 1.0},
    }))
    volume = native.Volume.open(str(root))
    np.testing.assert_array_equal(volume.read_zyx((0, 0, 0), (4, 4, 4)), expected(dtype))
    data, mask, stats = volume.read_zyx((-1, -1, -1), (6, 6, 6),
                                      with_mask=True, with_stats=True)
    np.testing.assert_array_equal(data[1:5, 1:5, 1:5], expected(dtype))
    assert not data[0].any()
    assert not mask[0].any()
    assert isinstance(stats, dict)
    # Coordinate API intentionally returns rendering uint8, including for u16.
    xyz = np.array(list(itertools.product([-.2, .25, 1.5, 2.75, 4.],
                                         repeat=3)), np.float32).reshape(5, 25, 3)
    valid = np.ones(xyz.shape[:2], dtype=bool)
    reference = volume.sample_coords(xyz, valid, sampler_threads=1,
                                     chunk_locality_fast_path=False)
    for threads in (1, 2):
        result = volume.sample_coords(xyz, valid, sampler_threads=threads,
                                      chunk_locality_fast_path=True)
        np.testing.assert_array_equal(result[0], reference[0])
        np.testing.assert_array_equal(result[1], reference[1])
    with pytest.raises(ValueError):
        volume.sample_coords(xyz, valid, sampler_threads=-1)


def test_normal_axis_sign_jump_and_validity(tmp_path):
    groups = {}
    for name, value in (("grad_mag", 1), ("nx", 255), ("ny", 128)):
        directory = tmp_path / (name + ".zarr")
        directory.mkdir()
        (directory / ".zarray").write_text(json.dumps({
            "zarr_format": 2, "shape": [8, 8, 8], "chunks": [4, 4, 4],
            "dtype": "|u1", "compressor": None, "fill_value": 0,
            "order": "C", "filters": None,
        }))
        data = np.full((4, 4, 4), value, dtype=np.uint8)
        if name == "nx":
            data[:, :, 2:] = 1  # Opposite signs encode the same unoriented axis.
        if name == "grad_mag":
            data[1, 1, 1] = 0
        (directory / "0.0.0").write_bytes(data.tobytes())
        groups[name] = {"zarr": directory.name, "scaledown": 0, "channels": [name]}
    manifest = tmp_path / "normals.lasagna.json"
    manifest.write_text(json.dumps({
        "version": 2, "source_to_base": 1, "base_shape_zyx": [8, 8, 8],
        "grad_mag_encode_scale": 1000, "grad_mag_factor": .25, "groups": groups,
    }))
    sampler = native.NormalSampler(str(manifest))
    points = np.array([[1.5, 2, 2], [1, 1, 1], [6, 6, 6], [2, 2, 2]], np.float32)
    normals, valid = sampler.sample(points)
    assert valid.tolist() == [True, False, False, True]
    np.testing.assert_allclose(np.abs(normals[valid]), [[1, 0, 0], [1, 0, 0]], atol=1e-5)
    parallel, parallel_valid = sampler.sample(points, threads=2)
    np.testing.assert_array_equal(parallel_valid, valid)
    np.testing.assert_array_equal(parallel, normals)
    empty, empty_valid = sampler.sample(np.empty((0, 3), np.float32))
    assert empty.shape == (0, 3) and empty_valid.shape == (0,)
    with pytest.raises(ValueError):
        sampler.sample(points, threads=0)


def test_real_volume_exact_locality(request):
    path = request.config.getoption("--vc-real-volume")
    if not path:
        pytest.skip("optional real-volume validation; supply --vc-real-volume")
    level = request.config.getoption("--vc-level")
    origin = np.array([float(v) for v in request.config.getoption("--vc-origin-xyz").split(",")],
                      dtype=np.float32)
    assert origin.shape == (3,) and 0 <= level <= 10
    volume = native.Volume.open(path)
    native.set_chunk_cache_budget(256 * 1024**2)
    native.set_chunk_cache_decode_threads(1)
    native.set_chunk_cache_io_threads(1)
    z, y, x = np.indices((128, 128, 128), dtype=np.float32)
    xyz = np.stack([x, y, z], axis=-1).reshape(128 * 128, 128, 3)
    xyz = np.ascontiguousarray(origin + (xyz + .25) * (2**level))
    valid = np.ones(xyz.shape[:2], dtype=bool)
    # Warm both paths; timed intervals measure resident sampling, not disk speed.
    expected_result = volume.sample_coords(xyz, valid, level=level, sampler_threads=1,
                                          chunk_locality_fast_path=False)
    assert np.count_nonzero(expected_result[0]) > 0, "crop contains no nonzero CT data"
    volume.sample_coords(xyz, valid, level=level, sampler_threads=1)
    timing = {False: [], True: []}
    stages = {False: [], True: []}
    for repeat in range(10):
        # Alternate order to reduce systematic warm-up/drift bias.
        for fast in ((False, True) if repeat % 2 == 0 else (True, False)):
            start = time.perf_counter()
            result = volume.sample_coords(xyz, valid, level=level, sampler_threads=1,
                                          chunk_locality_fast_path=fast)
            timing[fast].append(time.perf_counter() - start)
            np.testing.assert_array_equal(result[0], expected_result[0])
            np.testing.assert_array_equal(result[1], expected_result[1])
            assert result[2]["decoded_chunks"] == 0
            stages[fast].append(result[2])
    print(json.dumps({
        "volume": path, "origin_base_xyz": origin.tolist(), "level": level,
        "shape_zyx": [128] * 3, "sample_workers": 1, "iterations_per_path": 10,
        "binary": native.__file__, "nonzero_voxels": int(np.count_nonzero(expected_result[0])),
        "results": {
            ("locality" if fast else "scalar_reference"): {
                "mean_ms": 1000 * float(np.mean(values)),
                "p50_ms": 1000 * float(np.median(values)),
                "p95_ms": 1000 * float(np.percentile(values, 95)),
                "samples_per_second": 1 / float(np.mean(values)),
                "million_voxels_per_second": 128**3 / float(np.mean(values)) / 1e6,
                "native_stage_mean_seconds": {
                    key: float(np.mean([sample[key] for sample in stages[fast]]))
                    for key in ("dependency_seconds", "chunk_resolve_seconds", "sample_seconds")
                },
            } for fast, values in timing.items()
        },
    }, sort_keys=True))
