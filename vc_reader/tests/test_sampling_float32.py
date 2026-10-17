"""Regression checks for the experimental rebased float32 CT sampler."""
import itertools
import json

import numpy as np
import pytest

pytest.importorskip("vc_reader._native")
from vc_reader.reader import ZarrReader


def make_array(path, shift):
    path.mkdir()
    chunks = np.array([4, 5, 3])
    shape = np.array([8, 10, 9])
    (path / ".zarray").write_text(json.dumps(dict(zarr_format=2, shape=(shape+shift).tolist(),
        chunks=chunks.tolist(), dtype="|u1", compressor=None, fill_value=37, order="C", filters=None)))
    rng = np.random.default_rng(43)
    for idx in itertools.product(range(2),range(2),range(3)):
        if idx == (1,1,1):
            continue
        values = rng.integers(0,256,tuple(chunks),dtype=np.uint8)
        key = ".".join(map(str,np.array(idx)+shift//chunks))
        (path/key).write_bytes(values.tobytes())


def points32(reader, points, offset, order):
    out = np.empty(len(points),np.uint8 if order==0 else np.float32)
    reader._array._sample_u8_float32(points,list(map(int,offset)),out.view(np.uint8),order)
    return out


@pytest.mark.parametrize("order",[0,1])
def test_rebased_float32_translation(tmp_path,order):
    shift = np.array([1_000_000,2_000_000,3_000_000])
    make_array(tmp_path/"local",np.zeros(3,dtype=np.int64))
    make_array(tmp_path/"global",shift)
    rng = np.random.default_rng(4)
    points = rng.uniform([0,0,0],[7,9,8],(2000,3)).astype(np.float32)
    points = np.concatenate([points,np.array([[0,.5,1.5],[3.5,4.5,5.5],[4,5,3]],np.float32)])
    with ZarrReader.open(str(tmp_path/"local")) as local, ZarrReader.open(str(tmp_path/"global")) as global_:
        expected = points32(local,points,[0,0,0],order)
        np.testing.assert_array_equal(points32(global_,points,shift,order),expected)
        # An odd integer origin must not change ties-to-even in global space.
        exact_points = np.array([[0,.5,1.5],[3.5,4.5,5.5],[4,5,3]],np.float32)
        np.testing.assert_array_equal(points32(global_,exact_points-1,shift+1,order),
                                      points32(local,exact_points,[0,0,0],order))
        reference = local.sample(points.astype(np.float64),order=order,dtype=np.float64)
        np.testing.assert_allclose(expected,reference,atol=4e-5,rtol=1e-6)
        shape,origin,axes = [3,4,5],[.25,.5,.75],[.8,0,.1, 0,.9,.1, .1,0,1.]
        outputs = []
        for reader,offset in [(local,[0,0,0]),(global_,shift.tolist())]:
            out = np.empty(shape,np.uint8 if order==0 else np.float32)
            reader._array._affine_u8_float32(shape,origin,axes,offset,out.reshape(-1).view(np.uint8),order)
            outputs.append(out)
        np.testing.assert_array_equal(*outputs)


def test_float32_boundaries_and_nonfinite(tmp_path):
    make_array(tmp_path/"a",np.zeros(3,dtype=np.int64))
    points = np.array([[-1,0,0],[-.5,0,0],[7.5,9.5,8.5],[8,10,9],
                       [np.nan,0,0],[0,np.inf,0],[0,0,-np.inf]],np.float32)
    with ZarrReader.open(str(tmp_path/"a")) as reader:
        for order in (0,1):
            actual = points32(reader,points,[0,0,0],order)
            reference = reader.sample(points.astype(np.float64),order=order,dtype=np.float64)
            np.testing.assert_array_equal(actual,reference)
        with pytest.raises(ValueError,match="output size"):
            reader._array._sample_u8_float32(points,[0,0,0],np.empty(1,np.uint8),1)
        with pytest.raises(ValueError,match="origin"):
            reader._array._sample_u8_float32(points,[-1,0,0],np.empty(len(points)*4,np.uint8),1)
