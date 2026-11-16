import numpy as np
import pytest

import helirec

MODES = {"zero": "constant", "edge": "edge", "reflect": "reflect", "symmetric": "symmetric", "wrap": "wrap"}


def numpy_reference(stack, ndim, ph, pw, top, left, mode, filt):
    h, w = stack.shape[-2:]
    out = np.empty_like(stack)
    for i, frame in enumerate(stack):
        if ndim == 2:
            padded = np.pad(frame, ((top, ph - h - top), (left, pw - w - left)), mode=MODES[mode])
            res = np.fft.irfft2(np.fft.rfft2(padded) * filt, s=(ph, pw))
            out[i] = res[top : top + h, left : left + w]
        else:
            padded = np.pad(frame, ((0, 0), (left, pw - w - left)), mode=MODES[mode])
            res = np.fft.irfft(np.fft.rfft(padded, axis=-1) * filt, n=pw, axis=-1)
            out[i] = res[:, left : left + w]
    return out


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("ndim", [1, 2])
def test_fourier_filter_matches_numpy(device, mode, ndim):
    rng = np.random.default_rng(3)
    h, w, ph, pw, top, left = 23, 37, 64, 96, 9, 21
    stack = rng.standard_normal((3, h, w)).astype(np.float32)
    shape = (ph, pw // 2 + 1) if ndim == 2 else (pw // 2 + 1,)
    filt = rng.uniform(0, 2, shape).astype(np.float32)
    f = helirec.FourierFilter(ndim, h, w, ph, pw, top, left, mode, filt, device=device)
    out = f.apply(stack)
    ref = numpy_reference(stack.astype(np.float64), ndim, ph, pw, top, left, mode, filt)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_bad_filter_size():
    with pytest.raises(RuntimeError, match="filter has"):
        helirec.FourierFilter(2, 4, 4, 8, 8, 0, 0, "edge", np.ones(3, np.float32))
