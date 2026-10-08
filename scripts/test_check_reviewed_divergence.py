import json
import numpy as np
import numcodecs
from check_reviewed_divergence import read_region


def test_cross_chunk_decode_and_missing_mask(tmp_path):
    codec = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=1)
    expected = np.arange(8**3, dtype=np.uint16).reshape(8,8,8).astype(np.uint8)
    for channel in ('presence','nx','ny'):
        path = tmp_path/f'fiber_{channel}.ome.zarr/1'
        path.mkdir(parents=True)
        (path/'.zarray').write_text(json.dumps(dict(zarr_format=2, filters=None, order='C',
            chunks=[4,4,4], shape=[8,8,8], dtype='|u1', fill_value=0, dimension_separator='/', compressor=codec.get_config())))
        for z,y,x in np.ndindex(2,2,2):
            target = path/str(z)/str(y)/str(x)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(codec.encode(expected[z*4:z*4+4,y*4:y*4+4,x*4:x*4+4].copy()))
    arrays, valid = read_region(tmp_path,np.array([2,2,2]),np.array([6,6,6]))
    assert valid.all()
    for data in arrays.values():
        np.testing.assert_array_equal(data,expected[2:6,2:6,2:6])
    (tmp_path/'fiber_ny.ome.zarr/1/1/1/1').unlink()
    _, valid = read_region(tmp_path,np.array([2,2,2]),np.array([6,6,6]))
    assert valid.sum() == 56
    assert not valid[2:,2:,2:].any()
