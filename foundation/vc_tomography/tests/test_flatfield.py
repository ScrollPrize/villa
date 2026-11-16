import numpy as np
import pytest

import vc_tomography


def reference(raw, dark, flat, threshold=1e-4, take_log=True):
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (raw - dark) / (flat - dark)
    v = np.where(np.isnan(v), threshold, np.maximum(v, threshold)).astype(np.float32)
    return -np.log(v) if take_log else v


@pytest.mark.parametrize("take_log", [True, False])
def test_flatfield_matches_numpy(device, take_log):
    rng = np.random.default_rng(0)
    dark = rng.uniform(90, 110, (64, 96)).astype(np.float32)
    flat = dark + rng.uniform(500, 1500, (64, 96)).astype(np.float32)
    raw = dark + rng.uniform(-20, 1500, (7, 64, 96)).astype(np.float32)
    raw[0, 0, 0] = dark[0, 0]  # zero signal -> threshold
    flat[1, 1] = dark[1, 1]  # no flat signal: 0/0 (NaN) for raw == dark -> threshold, inf otherwise
    raw[2, 1, 1] = dark[1, 1]
    out = vc_tomography.flatfield(raw, dark, flat, threshold=1e-4, take_log=take_log, device=device)
    assert out.shape == raw.shape and out.dtype == np.float32
    np.testing.assert_allclose(out, reference(raw, dark, flat, 1e-4, take_log), rtol=2e-6, atol=2e-6)
