"""Scalar CUDA channel views share the packed source and keep spatial APIs."""
import os

import numpy as np
import pytest

from build_support import load_build
load_build(names=("_catalog", "_reader"))

from vc_reader.reader import ZarrReader
from vc_reader.reader.cuda import CudaPool, is_available


def test_real_compound_channels():
    location = os.environ.get("VC_READER_COMPOUND_CUDA_TEST_ARRAY")
    if not location or not is_available():
        pytest.skip("requires a CUDA grant and VC_READER_COMPOUND_CUDA_TEST_ARRAY")
    # Region surrounding the PHerc0332 annotated fiber, at physical level 2.
    start = (6528, 2000, 1536)
    selection = tuple(slice(s, s+8) for s in start)
    points = np.array([start, np.array(start)+[1.25, 2.5, 3.75]], np.float32)
    pool = CudaPool(cache_bytes=64 << 20)
    with ZarrReader.open(location) as reader:
        for channel in range(reader.shape[0]):
            expected = reader[(channel, *selection)]
            with reader.cuda(channel=channel, pool=pool) as gpu:
                for source in ["cpu", "chunks", "auto"]:
                    np.testing.assert_array_equal(gpu.read(selection, source=source).numpy(), expected)
                for order in [0, 1]:
                    a = gpu.sample(points, order=order, source="cpu").numpy()
                    b = gpu.sample(points, order=order, source="chunks").numpy()
                    np.testing.assert_allclose(a, b, rtol=0, atol=1.e-3)
                level = reader.grid_level(cuda=gpu)
                assert level.gpu and level.channel == channel
                assert gpu.stats["chunk_uploads"] > 0
        with pytest.raises(ValueError):
            reader.cuda(channel=reader.shape[0], pool=pool)
