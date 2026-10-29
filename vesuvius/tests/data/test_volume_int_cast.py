"""Volume(return_as_type=<int>) must not wrap or truncate."""
import numpy as np
import zarr

from vesuvius.data.volume import Volume


def _vol(tmp_path, data, **kw):
    g = zarr.open_group(str(tmp_path / "v.zarr"), mode="w")
    g.zeros(name="0", shape=data.shape, chunks=data.shape, dtype=data.dtype)[...] = data
    return Volume(type="zarr", path=str(tmp_path / "v.zarr"), verbose=False, **kw)


def test_uint16_to_uint8_rescales_instead_of_wrapping(tmp_path):
    data = np.array([0, 255, 256, 511, 32768, 65535], np.uint16).reshape(1, 2, 3)
    out = _vol(tmp_path, data, return_as_type="np.uint8")[0:1, 0:2, 0:3]
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, (data >> 8).astype(np.uint8))


def test_int16_to_uint8_clips(tmp_path):
    data = np.array([-5, 0, 100, 300], np.int16).reshape(1, 2, 2)
    out = _vol(tmp_path, data, return_as_type="np.uint8")[0:1, 0:2, 0:2]
    np.testing.assert_array_equal(out.ravel(), [0, 0, 100, 255])


def test_uint8_to_uint16_is_plain_cast(tmp_path):
    data = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    out = _vol(tmp_path, data, return_as_type="np.uint16")[0:2, 0:2, 0:2]
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, data)


def test_minmax_to_uint8_rounds(tmp_path):
    data = np.arange(1000, dtype=np.uint16).reshape(1, 1, -1)
    out = _vol(tmp_path, data, normalization_scheme="instance_minmax",
               return_as_type="np.uint8")[0:1, 0:1, 0:1000]
    scaled = (data.astype(np.float32) / np.float32(999)) * np.float32(255)
    np.testing.assert_array_equal(out, np.rint(scaled).astype(np.uint8))


# --- float -> wide int: must be computed in float64 (float32 can't hold 2**31-1 / 2**32-1) ---
# NaN policy: NaN -> 0; +inf -> target max; -inf -> target min.
import itertools
import pytest

_I32, _U32 = np.iinfo(np.int32), np.iinfo(np.uint32)


@pytest.mark.parametrize("tgt", [np.int32, np.uint32])
def test_minmax_endpoints_wide_int(tmp_path, tgt):
    data = np.array([0, 0.5, 1], np.float32).reshape(1, 1, 3)
    out = _vol(tmp_path, data, normalization_scheme="instance_minmax",
               return_as_type=f"np.{np.dtype(tgt).name}")[0:1, 0:1, 0:3].ravel()
    mx = np.iinfo(tgt).max
    assert out.dtype == tgt
    assert out[0] == 0 and out[2] == mx
    assert abs(int(out[1]) - mx // 2) <= 1


@pytest.mark.parametrize("tgt,val", [(np.int32, 2.0**31), (np.uint32, 2.0**32)])
def test_float32_saturates(tmp_path, tgt, val):
    data = np.array([val, -val], np.float32).reshape(1, 1, 2)
    out = _vol(tmp_path, data, return_as_type=f"np.{np.dtype(tgt).name}")[0:1, 0:1, 0:2].ravel()
    info = np.iinfo(tgt)
    np.testing.assert_array_equal(out, [info.max, info.min])


@pytest.mark.parametrize("tgt", [np.uint8, np.int16, np.int32, np.uint32])
@pytest.mark.parametrize("src", [np.float32, np.float64])
def test_nan_inf_policy(tmp_path, src, tgt):
    data = np.array([np.nan, np.inf, -np.inf, 1], src).reshape(1, 1, 4)
    out = _vol(tmp_path, data, return_as_type=f"np.{np.dtype(tgt).name}")[0:1, 0:1, 0:4].ravel()
    info = np.iinfo(tgt)
    np.testing.assert_array_equal(out, [0, info.max, info.min, 1])


_DT = [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.float32, np.float64]


def _extremes(dt):
    if np.issubdtype(dt, np.integer):
        i = np.iinfo(dt)
        return np.unique(np.array([i.min, i.min + 1, 0, 1, i.max - 1, i.max], dt))
    big = np.finfo(dt).max
    return np.array([-big, -2.0**33, -2.0**32, -2.0**31, -1, 0, 1, 2.0**31, 2.0**32, 2.0**33, big], dt)


@pytest.mark.parametrize("src,tgt", list(itertools.product(_DT, _DT)))
def test_sweep_no_wrap(tmp_path, src, tgt):
    data = _extremes(src).reshape(1, 1, -1)
    n = data.shape[-1]
    out = _vol(tmp_path, data, return_as_type=f"np.{np.dtype(tgt).name}")[0:1, 0:1, 0:n].ravel()
    assert out.dtype == tgt
    o = out.astype(np.float64)
    # Order-preserving (no wrap) and sign-consistent.
    assert np.all(np.diff(o) >= 0), (data.ravel(), out)
    s = data.ravel().astype(np.float64)
    if np.issubdtype(tgt, np.integer):
        info = np.iinfo(tgt)
        assert o[0] == max(info.min, min(s[0], info.max)) or np.issubdtype(src, np.unsignedinteger)
        if not (np.issubdtype(src, np.unsignedinteger) and np.issubdtype(tgt, np.unsignedinteger)):
            assert o[-1] == min(info.max, s[-1])
            assert np.all(o[s < 0] <= 0) and np.all(o[s > 0] >= 0)


# --- 64-bit integer targets: float64 cannot represent int64/uint64 max ---
@pytest.mark.parametrize("tgt", [np.int64, np.uint64])
def test_minmax_endpoints_64bit(tmp_path, tgt):
    data = np.array([0, 0.5, 1], np.float32).reshape(1, 1, 3)
    out = _vol(tmp_path, data, normalization_scheme="instance_minmax",
               return_as_type=f"np.{np.dtype(tgt).name}")[0:1, 0:1, 0:3].ravel()
    assert out.dtype == tgt
    assert int(out[0]) == 0 and int(out[2]) == int(np.iinfo(tgt).max)


@pytest.mark.parametrize("src", [np.float32, np.float64])
@pytest.mark.parametrize("tgt", [np.int64, np.uint64])
def test_float_to_64bit_saturates(tmp_path, src, tgt):
    info = np.iinfo(tgt)
    vals = [np.nan, np.inf, -np.inf, 2.0**64, -2.0**64, 2.0**63, -2.0**63,
            np.finfo(src).max, -np.finfo(src).max, 1.0, 2.0**53]
    data = np.array(vals, src).reshape(1, 1, -1)
    out = _vol(tmp_path, data, return_as_type=f"np.{np.dtype(tgt).name}")[0:1, 0:1, 0:len(vals)].ravel()
    hi, lo = int(info.max), int(info.min)

    def oracle(v):
        if np.isnan(v):
            return 0
        if v == np.inf:
            return hi
        if v == -np.inf:
            return lo
        return min(hi, max(lo, int(v)))
    assert [int(x) for x in out] == [oracle(float(v)) for v in data.ravel()]


@pytest.mark.parametrize("src", _DT[:6] + [np.int64, np.uint64])
@pytest.mark.parametrize("tgt", [np.int64, np.uint64])
def test_int_to_64bit_extremes(tmp_path, src, tgt):
    data = _extremes(src).reshape(1, 1, -1)
    n = data.shape[-1]
    out = _vol(tmp_path, data, return_as_type=f"np.{np.dtype(tgt).name}")[0:1, 0:1, 0:n].ravel()
    info = np.iinfo(tgt)
    assert out.dtype == tgt
    expected = [min(int(info.max), max(int(info.min), int(v))) for v in data.ravel()]
    assert [int(x) for x in out] == expected
