import pytest

import vc_tomography


@pytest.fixture(params=vc_tomography.devices())
def device(request):
    """Each device this build can use: "cpu", and "cuda" when a GPU is usable."""
    return request.param
