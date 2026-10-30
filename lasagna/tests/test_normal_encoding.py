import numpy as np
import pytest

from lasagna.normal_encoding import _encode_normal_nxny_u8, encode_normal_nxny_u8


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("strided", [False, True])
def test_cache_blocked_encoding_matches_whole_array(dtype, strided):
    values = np.random.default_rng(42).uniform(-0.1, 1.1, (6, 65, 513)).astype(dtype)
    if strided:
        values = values[:, :, ::2]
    for eps in (1e-12, 1e-6):
        expected = _encode_normal_nxny_u8(*values, eps=eps)
        actual = encode_normal_nxny_u8(*values, eps=eps)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)


def test_cache_blocking_preserves_broadcast_inputs():
    values = [np.full((65, 513), 0.75), *([np.asarray(0.5)] * 5)]
    for got, want in zip(encode_normal_nxny_u8(*values), _encode_normal_nxny_u8(*values)):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_cache_blocking_preserves_degenerate_directions(value):
    values = np.full((6, 65, 513), value, dtype=np.float32)
    for got, want in zip(encode_normal_nxny_u8(*values), _encode_normal_nxny_u8(*values)):
        np.testing.assert_array_equal(got, want)
