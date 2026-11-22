"""Shared pytest fixtures. The build in VC_READER_BUILD_DIR is selected before any test module imports vc_reader."""
import pytest

# On AMD, PyTorch ships its own libamdhip64. If vc_reader is loaded first,
# PyTorch asks for its copy under another name (without ".7"), so both copies
# are loaded and PyTorch fails. Importing PyTorch first keeps one shared copy.
try:
    import torch  # noqa: F401
except ImportError:
    pass

from build_support import load_build
load_build()


@pytest.fixture
def torch_cuda():
    from vc_reader import cuda
    torch = pytest.importorskip('torch')
    if not cuda.is_available() or not torch.cuda.is_available():
        pytest.skip('CUDA build/device required')
    return torch
