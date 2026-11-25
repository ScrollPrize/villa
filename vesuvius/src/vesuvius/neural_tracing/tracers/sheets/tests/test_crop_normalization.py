import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.tracers.shared.crop_sampling import normalization


def test_zscore_crops_cuda_without_native_backend_matches_cpu(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    monkeypatch.setattr(normalization.vc_cuda, 'is_available', lambda: False)
    rng = np.random.default_rng(7)
    raw = rng.normal(loc=12.0, scale=3.5, size=(2, 3, 4, 4, 4)).astype(np.float32)
    cpu_out = normalization.zscore_crops(torch.from_numpy(raw))
    cuda_out = normalization.zscore_crops(torch.from_numpy(raw).cuda())
    assert cuda_out.device.type == 'cuda'
    torch.testing.assert_close(cuda_out.cpu(), cpu_out)
