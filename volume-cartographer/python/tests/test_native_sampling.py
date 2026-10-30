"""NormalSampler binding regression; select a CMake build using --vc-build-python."""
import json

import numpy as np
import pytest
import vc.volume as native


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
