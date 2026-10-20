"""Native reader integration through the public object-store adapter."""
import json

import numpy as np
import pytest

from vc_object_store import CtSheetNormalSampler


def test_native_ct_adapter_levels_scaling_and_support(tmp_path):
    level = tmp_path / "coarse"
    level.mkdir()
    (tmp_path / ".zattrs").write_text(json.dumps({
        "multiscales": [{"datasets": [{"path": "coarse"}]}],
    }))
    (level / ".zarray").write_text(json.dumps({
        "zarr_format": 2, "shape": [109] * 3, "chunks": [109] * 3,
        "dtype": "|u1", "fill_value": 0, "order": "C",
        "compressor": None, "filters": None,
    }))
    wave = (120 + 70 * np.sin((np.arange(109) - 54) * .3)).astype(np.uint8)
    np.broadcast_to(wave, (109, 109, 109)).tofile(level / "0.0.0")
    sampler = CtSheetNormalSampler(tmp_path, working_to_ct_scale=.5)
    point = [110.5, 113., 108.25]
    sample = sampler.sample_normal(point)
    assert sample["valid"] and sample["gap"] > .99
    np.testing.assert_allclose(np.abs(sample["normal"]), [1, 0, 0], atol=1e-5)
    assert sampler.sample_normal(point) == sample
    assert not sampler.sample_normal([107.98, 108, 108])["valid"]
    with pytest.raises(ValueError):
        sampler.sample_normal([np.nan, 108, 108])
    with pytest.raises(ValueError):
        CtSheetNormalSampler(tmp_path, working_to_ct_scale=0)


def test_removed_lasagna_option_is_rejected(db):
    with pytest.raises((ValueError, TypeError, RuntimeError), match="normals_manifest"):
        db.link_fibers_to_patches(normals_manifest="unused.json")
