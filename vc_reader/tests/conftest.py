"""Shared pytest fixtures."""
import pytest


@pytest.fixture
def torch_cuda():
    from vc_reader import cuda
    torch = pytest.importorskip('torch')
    if not cuda.is_available() or not torch.cuda.is_available():
        pytest.skip('CUDA build/device required')
    return torch
