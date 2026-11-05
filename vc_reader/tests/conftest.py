"""Shared pytest fixtures. The build in VC_READER_BUILD_DIR is selected before any test module imports vc_reader."""
import pytest

from build_support import load_build
load_build()


@pytest.fixture
def torch_cuda():
    from vc_reader import cuda
    torch = pytest.importorskip('torch')
    if not cuda.is_available() or not torch.cuda.is_available():
        pytest.skip('CUDA build/device required')
    return torch
