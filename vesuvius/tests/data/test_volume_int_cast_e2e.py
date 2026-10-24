"""End-to-end Volume read path vs an independent Python-int oracle.

Every (source dtype, target dtype) pair is written to a small local zarr store
(v2, and v3 when zarr>=3 is installed), read back through ``Volume.__getitem__``
and compared element-wise with an oracle that uses only Python ints/Fractions.
"""
import math
from fractions import Fraction

import numpy as np
import pytest
import zarr

from vesuvius.data.volume import Volume

_ZARR_MAJOR = int(zarr.__version__.split(".")[0])
FORMATS = [2] + ([3] if _ZARR_MAJOR >= 3 else [])

INT_SRC = ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]
FLOAT_SRC = ["float16", "float32", "float64"]
INT_TGT = ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]
FLOAT_TGT = ["float32", "float64"]


def _src_values(src):
    dt = np.dtype(src)
    if dt.kind in "iu":
        i = np.iinfo(dt)
        vals = {i.min, i.min + 1, i.max, i.max - 1, 0, 1, 127, 128, 255, 256, 32767, 32768, 65535, 65536,
                2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**53 + 1, 2**63 - 1, -1, -128, -129, -32768, -32769, -2**31}
        return np.array(sorted(v for v in vals if i.min <= v <= i.max), dtype=dt)
    f = np.finfo(dt)
    vals = [0.0, -0.0, 0.4, 0.5, 1.5, 2.5, -0.5, -1.5, 127.5, 255.49, 255.5, 256.0, -128.5, 32767.5,
            65535.4, 2.0**31 - 1, 2.0**31, 2.0**32 - 1, 2.0**32, 2.0**53, 2.0**63, 2.0**64, -2.0**63,
            -2.0**63 - 4096, 1e30, -1e30, float(f.max), float(f.min), float(f.tiny),
            math.nan, math.inf, -math.inf]
    with np.errstate(over="ignore"):
        return np.array(vals, dtype=np.float64).astype(dt)


def _read(tmp_path, data, fmt, **kw):
    path = str(tmp_path / f"v{fmt}.zarr")
    if _ZARR_MAJOR >= 3:
        g = zarr.open_group(path, mode="w", zarr_format=fmt)
        a = g.create_array(name="0", shape=data.shape, chunks=data.shape, dtype=data.dtype)
    else:
        g = zarr.open_group(path, mode="w")
        a = g.zeros(name="0", shape=data.shape, chunks=data.shape, dtype=data.dtype)
    a[...] = data
    v = Volume(type="zarr", path=path, verbose=False, **kw)
    return v[0:1, 0:1, 0:data.shape[-1]].ravel()


def _round_half_even(q: Fraction) -> int:
    return round(q)  # Python's round on Fraction is exact banker's rounding


def _oracle_none(x, src, tgt):
    s, t = np.dtype(src), np.dtype(tgt)
    ti = np.iinfo(t)
    if s.kind in "iu":
        v = int(x)
        if s.kind == "u" and t.kind == "u" and np.iinfo(s).max > ti.max:
            return v >> (np.iinfo(s).bits - ti.bits)
        return min(max(v, ti.min), ti.max)
    f = float(x)
    if math.isnan(f):
        return 0
    if math.isinf(f):
        return ti.max if f > 0 else ti.min
    return min(max(_round_half_even(Fraction(f)), ti.min), ti.max)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tgt", INT_TGT)
@pytest.mark.parametrize("src", INT_SRC + FLOAT_SRC)
def test_no_normalization_int_target(tmp_path, fmt, src, tgt):
    data = _src_values(src).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, return_as_type=f"np.{tgt}")
    assert out.dtype == np.dtype(tgt)
    expected = [_oracle_none(x, src, tgt) for x in data.ravel()]
    assert [int(v) for v in out] == expected


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tgt", FLOAT_TGT)
@pytest.mark.parametrize("src", INT_SRC + FLOAT_SRC)
def test_no_normalization_float_target(tmp_path, fmt, src, tgt):
    data = _src_values(src).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, return_as_type=f"np.{tgt}")
    assert out.dtype == np.dtype(tgt)
    with np.errstate(over="ignore"):
        np.testing.assert_array_equal(out, data.ravel().astype(tgt))


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("src", INT_SRC + FLOAT_SRC)
def test_no_normalization_no_target_is_identity(tmp_path, fmt, src):
    data = _src_values(src).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt)
    assert out.dtype == data.dtype
    np.testing.assert_array_equal(out, data.ravel())


def _minmax_tolerance(tgt, value):
    # <= 1 unit (half-way cases computed in float64) for <=32-bit targets; for
    # 64-bit targets float64 cannot represent the result, allow one float64 ulp.
    if np.iinfo(tgt).bits < 64:
        return 1
    return max(1, int(np.spacing(np.float64(abs(value)))))


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tgt", INT_TGT)
@pytest.mark.parametrize("src", INT_SRC + FLOAT_SRC)
def test_instance_minmax_int_target(tmp_path, fmt, src, tgt):
    vals = _src_values(src)
    if vals.dtype.kind == "f":
        # keep a finite range float32 can subtract without overflow
        vals = vals[np.isfinite(vals) & (np.abs(vals.astype(np.float64)) <= 2.0**64)]
    data = vals.reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_minmax", return_as_type=f"np.{tgt}")
    assert out.dtype == np.dtype(tgt)
    xs = [Fraction(int(x)) if vals.dtype.kind in "iu" else Fraction(float(x)) for x in vals]
    lo, hi = min(xs), max(xs)
    ti = np.iinfo(tgt)
    for x, got in zip(xs, out):
        exact = _round_half_even((x - lo) / (hi - lo) * ti.max)
        assert 0 <= int(got) <= ti.max
        # integer sources are scaled with exact integer arithmetic
        tol = 0 if vals.dtype.kind in "iu" else _minmax_tolerance(tgt, exact)
        assert abs(int(got) - exact) <= tol, (x, int(got), exact)
    assert int(out[int(np.argmin(xs))]) == 0
    assert int(out[int(np.argmax(xs))]) == ti.max


# Concrete witnesses of the float32 limit/scaling bug (fail on 79ed0d4).
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("src,vals,tgt,idx,exact", [
    # uint32 2**32-2 is not the max but float32 rounds it to 2**32 -> saturated to max
    ("uint32", [0, 1, 2**32 - 2, 2**32 - 1], "uint32", 2, 2**32 - 2),
    # float32 [0,1] quantization: 128/255 * (2**32-1) off by 127 units
    ("uint8", list(range(256)), "uint32", 128, 2155905152),
    ("uint8", list(range(256)), "int32", 128, 1077952576),
    # int32 -1 lands exactly on a float32 tie: off by one even for uint8
    ("int32", [-2**31, -1, 0, 1, 2**31 - 1], "uint8", 1, 127),
])
def test_instance_minmax_witnesses(tmp_path, fmt, src, vals, tgt, idx, exact):
    data = np.array(vals, dtype=src).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_minmax", return_as_type=f"np.{tgt}")
    assert int(out[idx]) == exact


@pytest.mark.parametrize("fmt", FORMATS)
def test_instance_minmax_uint64_witness(tmp_path, fmt):
    # float32 [0,1] scaled to uint64: was off by 2**39 units
    data = np.arange(256, dtype=np.uint8).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_minmax", return_as_type="np.uint64")
    exact = round(Fraction(128, 255) * (2**64 - 1))
    assert abs(int(out[128]) - exact) <= 2048


@pytest.mark.parametrize("fmt", FORMATS)
def test_minmax_float_output_stays_float32(tmp_path, fmt):
    data = np.arange(10, dtype=np.uint16).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_minmax")
    assert out.dtype == np.float32
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_minmax", return_as_type="np.float32")
    assert out.dtype == np.float32


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tgt", INT_TGT)
def test_zscore_int_target_stays_float32(tmp_path, fmt, tgt):
    data = np.arange(10, dtype=np.uint16).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_zscore", return_as_type=f"np.{tgt}")
    assert out.dtype == np.float32


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tgt", INT_TGT)
def test_percentile_minmax_int_target_in_range(tmp_path, fmt, tgt):
    data = np.arange(1000, dtype=np.uint32).reshape(1, 1, -1) * 4000
    out = _read(tmp_path, data, fmt, normalization_scheme="percentile_minmax", return_as_type=f"np.{tgt}")
    ti = np.iinfo(tgt)
    assert out.dtype == np.dtype(tgt)
    assert int(out[0]) == 0 and int(out[-1]) == ti.max
    assert np.all(np.diff(out.astype(np.float64)) >= 0)


# --- REDTEAM-020: 64-bit integer inputs must not collapse before min/max ---
_U64MAX, _I64MIN, _I64MAX = 2**64 - 1, -2**63, 2**63 - 1


def _minmax_oracle(vals, tgt):
    xs = [Fraction(v) for v in vals]
    lo, hi = min(xs), max(xs)
    m = np.iinfo(tgt).max
    return [0 if hi == lo else round((x - lo) / (hi - lo) * m) for x in xs]


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tgt", ["uint8", "uint32", "uint64"])
@pytest.mark.parametrize("scheme", ["instance_minmax", "percentile_minmax"])
@pytest.mark.parametrize("src,vals", [
    ("uint64", [_U64MAX - 1, _U64MAX]),
    ("uint64", [_U64MAX - 2, _U64MAX - 1, _U64MAX]),
    ("uint64", [0, 1, 2]),
    ("int64", [_I64MAX - 1, _I64MAX]),
    ("int64", [_I64MIN, _I64MIN + 1, _I64MIN + 2]),
    ("int64", [_I64MIN, -1, 0, _I64MAX]),
    ("uint64", [0, 2**63, _U64MAX]),
])
def test_minmax_64bit_close_values_keep_contrast(tmp_path, fmt, tgt, scheme, src, vals):
    data = np.array(vals, dtype=src).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme=scheme, return_as_type=f"np.{tgt}")
    if scheme == "percentile_minmax":
        # percentile endpoints are interpolated; endpoints and order are exact
        assert int(out[0]) == 0 and int(out[-1]) == np.iinfo(tgt).max
        assert np.all(np.diff(out.astype(np.float64)) >= 0)
        return
    assert [int(v) for v in out] == _minmax_oracle(vals, tgt)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tgt", ["uint8", "uint64"])
@pytest.mark.parametrize("src", ["float32", "float64"])
def test_minmax_finite_range_wider_than_float(tmp_path, fmt, tgt, src):
    big = float(np.finfo(src).max) / 1.5
    data = np.array([-big, 0.0, big], dtype=src).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_minmax", return_as_type=f"np.{tgt}")
    m = np.iinfo(tgt).max
    assert int(out[0]) == 0 and int(out[2]) == m
    assert abs(int(out[1]) - round(Fraction(m, 2))) <= _minmax_tolerance(np.dtype(tgt), m // 2)


@pytest.mark.parametrize("fmt", FORMATS)
def test_minmax_multichannel_is_shifted_per_channel(tmp_path, fmt):
    path = str(tmp_path / f"c{fmt}.zarr")
    data = np.array([[_U64MAX - 1, _U64MAX], [0, 7]], dtype=np.uint64).reshape(2, 1, 1, 2)
    if _ZARR_MAJOR >= 3:
        a = zarr.open_group(path, mode="w", zarr_format=fmt).create_array(
            name="0", shape=data.shape, chunks=data.shape, dtype=data.dtype)
    else:
        a = zarr.open_group(path, mode="w").zeros(name="0", shape=data.shape, chunks=data.shape, dtype=data.dtype)
    a[...] = data
    v = Volume(type="zarr", path=path, verbose=False, normalization_scheme="instance_minmax",
               return_as_type="np.uint8")
    out = v[0:2, 0:1, 0:1, 0:2]
    np.testing.assert_array_equal(out.reshape(2, 2), [[0, 255], [0, 255]])


# Non-finite input, policy "finite_only" (REDTEAM-020 FOUND #3): statistics over the
# finite voxels; NaN -> 0, +inf -> top of range, -inf -> bottom; no finite voxel -> 0 + one warning.
_SCHEMES3 = ["instance_minmax", "percentile_minmax", "instance_zscore"]


def _read_any(tmp_path, data, fmt, name, **kw):
    sub = tmp_path / name
    sub.mkdir()
    return _read(sub, data, fmt, **kw)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("scheme", _SCHEMES3)
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("rt", [None, "np.float32", "np.uint8", "np.uint16"])
def test_nonfinite_voxel_does_not_poison_instance(tmp_path, fmt, scheme, bad, rt):
    clean = np.linspace(10, 110, 64, dtype=np.float32)
    data = clean.copy()
    data[20] = bad
    kw = dict(normalization_scheme=scheme)
    if rt:
        kw["return_as_type"] = rt
    out = _read_any(tmp_path, data.reshape(1, 1, -1), fmt, "a", **kw)
    keep = np.ones(64, bool)
    keep[20] = False
    ref = _read_any(tmp_path, clean[keep].reshape(1, 1, -1), fmt, "b", **kw)
    np.testing.assert_array_equal(out[keep], ref)  # finite voxels: as if the bad one were absent
    assert np.all(np.isfinite(out.astype(np.float64)))
    if scheme == "instance_zscore":
        want = {"nan": 0.0, "inf": ref.max(), "-inf": ref.min()}[str(bad)]
    else:
        top = 1.0 if rt in (None, "np.float32") else np.iinfo(rt.replace("np.", "")).max
        want = {"nan": 0, "inf": top, "-inf": 0}[str(bad)]
    assert out[20] == want


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("scheme", _SCHEMES3)
@pytest.mark.parametrize("rt", [None, "np.uint8"])
def test_all_nonfinite_instance_is_zero_with_one_warning(tmp_path, fmt, scheme, rt):
    data = np.array([math.nan, math.inf, -math.inf, math.nan], np.float32).reshape(1, 1, -1)
    kw = dict(normalization_scheme=scheme)
    if rt:
        kw["return_as_type"] = rt
    with pytest.warns(RuntimeWarning, match="no finite") as rec:
        out = _read(tmp_path, data, fmt, **kw)
    assert len([w for w in rec if "no finite" in str(w.message)]) == 1
    np.testing.assert_array_equal(out, [0, 0, 0, 0])


@pytest.mark.parametrize("fmt", FORMATS)
def test_nonfinite_with_huge_finite_range(tmp_path, fmt):
    data = np.array([-1e38, 0, 1e38, math.nan, math.inf], np.float32).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_minmax", return_as_type="np.uint8")
    np.testing.assert_array_equal(out, [0, 128, 255, 0, 255])


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("scheme", ["instance_minmax", "percentile_minmax"])
def test_minmax_2d_read_64bit_keeps_contrast(tmp_path, fmt, scheme):
    # 2D read (volume[z, :, :]): whichever grouping normalization uses, the
    # top-of-range uint64 values must keep their contrast.
    data = np.array([[_U64MAX - 3, _U64MAX - 2, _U64MAX - 1, _U64MAX]] * 2, dtype=np.uint64).reshape(1, 2, 4)
    path = str(tmp_path / f"d{fmt}.zarr")
    if _ZARR_MAJOR >= 3:
        a = zarr.open_group(path, mode="w", zarr_format=fmt).create_array(
            name="0", shape=data.shape, chunks=data.shape, dtype=data.dtype)
    else:
        a = zarr.open_group(path, mode="w").zeros(name="0", shape=data.shape, chunks=data.shape, dtype=data.dtype)
    a[...] = data
    v = Volume(type="zarr", path=path, verbose=False, normalization_scheme=scheme, return_as_type="np.uint8")
    out = v[0, 0:2, 0:4]
    assert out.shape == (2, 4)
    for row in out:
        assert int(row[0]) == 0 and int(row[-1]) == 255
        assert np.all(np.diff(row.astype(np.int64)) > 0)


# Off-by-one from float64 scaling of a 64-bit span (Ing X oracle, case i64_full_mid):
# -1 in [-2**63, 2**63-1] is 127.4999... of 255, so exact rounding gives 127, not 128.
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tgt", INT_TGT)
@pytest.mark.parametrize("src,vals", [
    ("int64", [_I64MIN, -1, 0, _I64MAX]),
    ("uint64", [0, 2**63, _U64MAX]),
    ("uint64", [_U64MAX - 2, _U64MAX - 1, _U64MAX]),
    ("uint64", [0, 2**53, 2**53 + 1, _U64MAX]),
    ("int64", [_I64MIN, _I64MIN + 2**62 + 1, 3, _I64MAX - 5, _I64MAX]),
    ("uint32", [0, 1, 2**31 - 1, 2**31, 2**32 - 2, 2**32 - 1]),
    ("int32", [-2**31, -1, 0, 1, 2**31 - 1]),
    ("uint16", [0, 1, 257, 32767, 32768, 65534, 65535]),
    ("uint8", list(range(256))),
])
def test_instance_minmax_integer_rounding_is_exact(tmp_path, fmt, tgt, src, vals):
    data = np.array(vals, dtype=src).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_minmax", return_as_type=f"np.{tgt}")
    assert out.dtype == np.dtype(tgt)
    assert [int(v) for v in out] == _minmax_oracle(vals, tgt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_instance_minmax_int64_off_by_one_witness(tmp_path, fmt):
    data = np.array([_I64MIN, -1, 0, _I64MAX], dtype=np.int64).reshape(1, 1, -1)
    for tgt, want in (("uint8", [0, 127, 128, 255]), ("uint16", [0, 32767, 32768, 65535])):
        out = _read(tmp_path, data, fmt, normalization_scheme="instance_minmax", return_as_type=f"np.{tgt}")
        assert [int(v) for v in out] == want


# REDTEAM-020 addendum: 64-bit TARGETS from ordinary uint8 data lost hundreds of
# levels in float64 (uint64 middle off by 341, int64 by 170).
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tgt", ["int64", "uint64"])
def test_instance_minmax_exact_64bit_target_interior(tmp_path, fmt, tgt):
    data = np.array([0, 1, 3], dtype=np.uint8).reshape(1, 1, 3)
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_minmax", return_as_type=f"np.{tgt}")
    top = int(np.iinfo(tgt).max)
    assert [int(v) for v in out] == [0, round(Fraction(top, 3)), top]


# REDTEAM-022: np.percentile([-1e308, 1e308], (1, 99)) overflows to [inf, -inf], which
# sent percentile_minmax to the all-zero branch. Real zarr2 Volume, shape (1, 1, n).
@pytest.mark.parametrize("vals,dtype,want", [
    ([-1e308, 1e308], np.float64, [0, 1]),
    ([-3e38, 3e38], np.float32, [0, 1]),
    ([-3e38, 3e38, math.inf], np.float32, [0, 1, 1]),
    ([-1e308, 1e308, math.inf], np.float64, [0, 1, 1]),
])
@pytest.mark.parametrize("rt", [None, "np.float32", "np.float64"])
def test_percentile_minmax_opposite_extremes(tmp_path, vals, dtype, want, rt):
    kw = dict(normalization_scheme="percentile_minmax")
    if rt:
        kw["return_as_type"] = rt
    data = np.array(vals, dtype).reshape(1, 1, -1)
    out = _read(tmp_path, data, 2, **kw)
    np.testing.assert_allclose(out.astype(np.float64), want, rtol=0, atol=1e-6)


@pytest.mark.parametrize("scheme", ["instance_minmax", "percentile_minmax"])
@pytest.mark.parametrize("rt,top", [("np.uint64", 2**64 - 1), ("np.int64", 2**63 - 1),
                                    ("np.uint8", 255)])
def test_bool_source_to_integer_targets(tmp_path, scheme, rt, top):
    data = np.array([True, False, True], bool).reshape(1, 1, -1)
    out = _read(tmp_path, data, 2, normalization_scheme=scheme, return_as_type=rt)
    assert out.dtype == np.dtype(rt.replace("np.", ""))
    assert out.tolist() == [top, 0, top]


# Overflow in z-score statistics and in float64 -> float32 work casts (real Volume).
_S2 = math.sqrt(2)


@pytest.mark.parametrize("vals,dtype,want", [
    ([-3e38, 3e38], np.float32, [-1, 1]),
    ([3e38, 3e38, 0], np.float32, [1 / _S2, 1 / _S2, -_S2]),
    ([-1e308, 1e308], np.float64, [-1, 1]),
    ([-1e200, 1e200], np.float64, [-1, 1]),
    ([1e308, 1e308], np.float64, [0, 0]),
    ([-3e38, 3e38, math.inf], np.float32, [-1, 1, 1]),
])
@pytest.mark.parametrize("fmt", FORMATS)
def test_instance_zscore_no_overflow(tmp_path, fmt, vals, dtype, want):
    data = np.array(vals, dtype).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_zscore")
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out.astype(np.float64).ravel(), want, rtol=0, atol=1e-6)


@pytest.mark.parametrize("vals,want", [
    ([0, 1e300, 5e299], [0, 1, 0.5]),
    ([-1e308, 1e308, 0], [0, 1, 0.5]),
])
@pytest.mark.parametrize("scheme", ["instance_minmax", "percentile_minmax"])
@pytest.mark.parametrize("fmt", FORMATS)
def test_float64_cast_overflow_keeps_float64_work(tmp_path, fmt, vals, want, scheme):
    data = np.array(vals, np.float64).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme=scheme)
    want = want if scheme == "instance_minmax" else [0, 1, 0.5]
    np.testing.assert_allclose(out.astype(np.float64).ravel(), want, rtol=0, atol=1e-6)


@pytest.mark.parametrize("fmt", FORMATS)
def test_float64_cast_underflow_keeps_float64_work(tmp_path, fmt):
    # 1e-300 flushes to 0 in float32, merging distinct values. The work stays float64;
    # the result is still tiny because the existing 1e-8 range/std floor applies.
    data = np.array([1e-300, 2e-300, 3e-300], np.float64).reshape(1, 1, -1)
    for scheme in ("instance_minmax", "instance_zscore"):
        out = _read(tmp_path / scheme, data, fmt, normalization_scheme=scheme).ravel()
        assert out.dtype == np.float64
        assert out[0] < out[1] < out[2]


@pytest.mark.parametrize("fmt", FORMATS)
def test_global_zscore_and_ct_no_overflow(tmp_path, fmt):
    data = np.array([3e38, -3e38], np.float32).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme="global_zscore",
                global_mean=-1e38, global_std=1.0)
    np.testing.assert_allclose(out.astype(np.float64).ravel(), [4e38, -2e38], rtol=1e-6)
    out = _read(tmp_path / "ct", data, fmt, normalization_scheme="ct", intensity_props={
        "mean": -1e38, "std": 1.0, "percentile_00_5": -3e38, "percentile_99_5": 3e38})
    np.testing.assert_allclose(out.astype(np.float64).ravel(), [4e38, -2e38], rtol=1e-6)


@pytest.mark.parametrize("fmt", FORMATS)
def test_percentile_minmax_rejects_bad_percentiles(tmp_path, fmt, monkeypatch):
    """The finite/ordered check is a real check (not an assert removed by -O)."""
    import vesuvius.data.volume as vol
    monkeypatch.setattr(vol.np, "percentile", lambda a, q, *args, **kw: np.array([np.inf, -np.inf]))
    data = np.array([0.0, 1.0, 2.0], np.float32).reshape(1, 1, -1)
    with pytest.raises(ValueError, match="not finite and ordered"):
        _read(tmp_path, data, fmt, normalization_scheme="percentile_minmax")


# FOUND C: finite_only z-score where the finite part needs scaling ([3e38, 3e38] overflows
# the float32 mean) and is constant: no NaN from the finite z min/max.
@pytest.mark.parametrize("bad,want_bad", [(math.inf, 0.0), (math.nan, 0.0), (-math.inf, 0.0)])
@pytest.mark.parametrize("fmt", FORMATS)
def test_instance_zscore_scaled_constant_with_nonfinite(tmp_path, fmt, bad, want_bad):
    data = np.array([3e38, 3e38, bad], np.float32).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_zscore").ravel()
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out, [0, 0, want_bad])


# FOUND B: float32 [-1e20, 1e20]: the float32 variance (1e40) overflows.
@pytest.mark.parametrize("fmt", FORMATS)
def test_instance_zscore_float32_variance_overflow(tmp_path, fmt):
    data = np.array([-1e20, 1e20], np.float32).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_zscore").ravel()
    np.testing.assert_allclose(out, [-1, 1], rtol=0, atol=1e-6)


# Normalization writes in place; it must never write into data the caller owns.
@pytest.mark.parametrize("scheme,rt", [
    ("instance_minmax", "none"), ("instance_minmax", "np.uint8"), ("percentile_minmax", "np.float32"),
    ("instance_zscore", "np.float32")])
@pytest.mark.parametrize("src", ["float32", "uint16"])
@pytest.mark.parametrize("backing", ["zarr", "memory_zarr", "numpy"])
def test_read_does_not_mutate_source(tmp_path, backing, src, scheme, rt):
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((2, 4, 8, 8)) * 100 + 300).astype(src)
    ref = data.copy()
    path = tmp_path / "v.zarr"
    g = zarr.open_group(str(path), mode="w")
    if _ZARR_MAJOR >= 3:
        g.create_array(name="0", shape=data.shape, chunks=(1, 4, 4, 4), dtype=data.dtype)[...] = data
    else:
        g.zeros(name="0", shape=data.shape, chunks=(1, 4, 4, 4), dtype=data.dtype)[...] = data
    vol = Volume(type="zarr", path=str(path), verbose=False, normalization_scheme=scheme, return_as_type=rt)
    if backing == "memory_zarr":
        mem = zarr.zeros(shape=data.shape, chunks=(1, 4, 4, 4), dtype=data.dtype)
        mem[...] = data
        vol.data = mem
    elif backing == "numpy":
        vol.data = [data]  # a level that is a plain ndarray
    raw = Volume(type="zarr", path=str(path), verbose=False, normalization_scheme="none")
    if backing == "numpy":
        raw.data = [data]
    keep = raw[:, :, :, :]
    keep_copy = np.array(keep, copy=True)
    first = np.array(vol[:, :, :, :], copy=True)
    second = vol[:, :, :, :]
    np.testing.assert_array_equal(second, first)
    np.testing.assert_array_equal(keep, keep_copy)  # an earlier raw read is untouched
    np.testing.assert_array_equal(data, ref)


# REDTEAM-023: x - mean overflows even in float64 (1e308 - (-1e308)); the representable
# result (1e308 + 1e308) / 1e308 = 2 must come back finite.
@pytest.mark.parametrize("scheme", ["global_zscore", "ct"])
@pytest.mark.parametrize("fmt", FORMATS)
def test_representable_float64_normalization(tmp_path, fmt, scheme):
    data = np.full((1, 1, 1), 1e308, np.float64)
    kw = ({"global_mean": -1e308, "global_std": 1e308} if scheme == "global_zscore" else
          {"intensity_props": {"mean": -1e308, "std": 1e308,
                               "percentile_00_5": -1e308, "percentile_99_5": 1e308}})
    out = _read(tmp_path, data, fmt, normalization_scheme=scheme, return_as_type="np.float64", **kw)
    np.testing.assert_allclose(out, [2.0])


def _gz_ct_kw(scheme, mean, std, lb=-np.finfo(np.float64).max, ub=np.finfo(np.float64).max):
    if scheme == "global_zscore":
        return {"global_mean": mean, "global_std": std}
    return {"intensity_props": {"mean": mean, "std": std, "percentile_00_5": lb, "percentile_99_5": ub}}


# The overflow path computes ((x/2 - mean/2) / std) * 2. Exact equality pins both halvings
# and the doubling: dropping any of them gives inf or a result off by a factor of 2.
@pytest.mark.parametrize("scheme", ["global_zscore", "ct"])
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("x, mean, std, expected", [
    (1e308, -1e308, 1e308, 2.0),
    (1.5e308, -1e308, 1e308, 2.5),
    (-1e308, 1e308, 1e308, -2.0),
    (1e308, -1e308, 4.0, 5e307),
])
def test_variance_overflow_path_halving_is_exact(tmp_path, fmt, scheme, x, mean, std, expected):
    data = np.full((1, 1, 1), x, np.float64)
    out = _read(tmp_path, data, fmt, normalization_scheme=scheme, return_as_type="np.float64",
                **_gz_ct_kw(scheme, mean, std))
    np.testing.assert_array_equal(out, [expected])


# Halving is exact only in the normal range: x * 0.5 loses the last bit of a subnormal
# (5e-324 * 0.5 -> 0.0). Non-overflowing voxels must keep the plain (x - mean) / std.
@pytest.mark.parametrize("scheme", ["global_zscore", "ct"])
@pytest.mark.parametrize("fmt", FORMATS)
def test_no_overflow_subnormal_inputs_skip_the_halving_path(tmp_path, fmt, scheme):
    data = np.array([[[5e-324, 1.5e-323, -5e-324]]], np.float64)
    out = _read(tmp_path, data, fmt, normalization_scheme=scheme, return_as_type="np.float64",
                **_gz_ct_kw(scheme, 0.0, 1.0))
    np.testing.assert_array_equal(out, [5e-324, 1.5e-323, -5e-324])
    assert out[2] != 0.0 and np.signbit(out[2])  # -5e-324, not -0.0


def _read_4d(tmp_path, data, fmt, **kw):
    path = str(tmp_path / f"v4d{fmt}.zarr")
    if _ZARR_MAJOR >= 3:
        a = zarr.open_group(path, mode="w", zarr_format=fmt).create_array(
            name="0", shape=data.shape, chunks=data.shape, dtype=data.dtype)
    else:
        a = zarr.open_group(path, mode="w").zeros(name="0", shape=data.shape, chunks=data.shape, dtype=data.dtype)
    a[...] = data
    v = Volume(type="zarr", path=path, verbose=False, **kw)
    return np.asarray(v[0:data.shape[0], 0:data.shape[1], 0:data.shape[2], 0:data.shape[3]])


# Array-valued global_mean is not a supported per-channel API, but it broadcasts like
# numpy, and the overflow path must broadcast the same way instead of raising.
@pytest.mark.parametrize("fmt", FORMATS)
def test_overflow_path_with_c111_mean(tmp_path, fmt):
    data = np.array([[1e308, 0.0], [1.0, 2.0], [-1e308, 5.0]], np.float64).reshape(3, 1, 1, 2)
    mean = np.array([-1e308, 0.0, 1e308]).reshape(3, 1, 1, 1)
    out = _read_4d(tmp_path, data, fmt, normalization_scheme="global_zscore", return_as_type="np.float64",
                   global_mean=mean, global_std=1e308)
    expected = np.array([[2.0, 1.0], [1.0 / 1e308, 2.0 / 1e308], [-2.0, (5.0 - 1e308) / 1e308]]).reshape(3, 1, 1, 2)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("fmt", FORMATS)
def test_overflow_path_with_list_mean(tmp_path, fmt):
    # A list mean broadcasts along the last axis (X), as on main.
    data = np.array([[[1e308, 1.0, 2.0]]], np.float64)
    out = _read(tmp_path, data, fmt, normalization_scheme="global_zscore", return_as_type="np.float64",
                global_mean=[-1e308, 0.0, 0.0], global_std=1e308)
    np.testing.assert_array_equal(out, [2.0, 1.0 / 1e308, 2.0 / 1e308])


# A per-channel mean as a list/tuple/(C,) array broadcasts along X, not C. When X != C
# the plain formula changes the shape; on the overflow path that must not come back
# silently as a (C, Z, Y, C) array: either the input shape or a clear ValueError.
@pytest.mark.parametrize("form", ["list", "tuple", "np(C,)", "list_np"])
@pytest.mark.parametrize("fmt", FORMATS)
def test_overflow_path_keeps_input_shape_for_vector_mean(tmp_path, fmt, form):
    data = np.array([[1e308, -1e308], [-1e308, 1e308]], np.float64).reshape(2, 2, 1, 1)
    m = [-1e308, -5e307]
    mean = {"list": m, "tuple": tuple(m), "np(C,)": np.array(m), "list_np": [np.float64(x) for x in m]}[form]
    try:
        out = _read_4d(tmp_path, data, fmt, normalization_scheme="global_zscore",
                       return_as_type="np.float64", global_mean=mean, global_std=1e308)
    except ValueError as e:
        assert "shape" in str(e)
    else:
        assert out.shape == data.shape


# REDTEAM-025: global_zscore must keep NumPy's promotion of the std, as on main: a
# float32 source with a float64 (or 0-d float64 array) std gives float64, and
# 1e-20 / 1e30 stays ~1e-50 instead of underflowing to 0 in float32.
@pytest.mark.parametrize("std", [np.float64(1e30), np.array(1e30, dtype=np.float64),
                                 np.int64(10**15), np.array(10**15, dtype=np.int64)],
                         ids=["f64", "0d_f64", "i64", "0d_i64"])
@pytest.mark.parametrize("fmt", FORMATS)
def test_global_zscore_preserves_numpy_scalar_promotion(tmp_path, fmt, std):
    x = np.array([[[1e-20, 1.0]]], np.float32)
    expected = (x - 0.0) / max(std, 1e-8)
    assert expected.dtype == np.float64
    path = str(tmp_path / f"p{fmt}.zarr")
    if _ZARR_MAJOR >= 3:
        a = zarr.open_group(path, mode="w", zarr_format=fmt).create_array(
            name="0", shape=x.shape, chunks=x.shape, dtype=x.dtype)
    else:
        a = zarr.open_group(path, mode="w").zeros(name="0", shape=x.shape, chunks=x.shape, dtype=x.dtype)
    a[...] = x
    v = Volume(type="zarr", path=path, verbose=False, normalization_scheme="global_zscore",
               return_as_type="none", global_mean=0.0, global_std=std)
    actual = np.asarray(v[0:1, 0:1, 0:2])
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


# int64 with one outlier at -2**63: after the exact min shift, 0..998 sit near 2**63, where
# float64 cannot tell them apart, so p1 == p99 and the whole instance came back 0.
@pytest.mark.parametrize("src", ["int64", "uint64"])
@pytest.mark.parametrize("fmt", FORMATS)
def test_percentile_minmax_wide_integer_span(tmp_path, fmt, src):
    base = np.arange(999)
    outlier = np.iinfo(src).min if src == "int64" else np.iinfo(src).max
    # Construct directly in the integer dtype: concatenation can promote uint64 max to float64.
    data = np.array([int(outlier)] + base.tolist(), dtype=src).reshape(1, 1, -1)
    assert int(data[0, 0, 0]) == int(outlier)
    out = _read(tmp_path, data, fmt, normalization_scheme="percentile_minmax", return_as_type="np.float64")
    vals = data.ravel().astype(object)
    s = sorted(vals)
    def pct(q):
        h = Fraction(len(s) - 1) * Fraction(q) / 100
        i = int(h)
        return Fraction(int(s[i])) + (h - i) * (int(s[i + 1]) - int(s[i]))
    lo, hi = pct(1), pct(99)
    expected = [float((min(max(Fraction(int(v)), lo), hi) - lo) / (hi - lo)) for v in vals]
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)  # float32 work dtype
    assert 0.07 < out[100] < 0.1
