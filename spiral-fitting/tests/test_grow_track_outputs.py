"""The tifxyz directories the track growers write."""
import json

import numpy as np
from vc_reader import tifxyz

from grow_track_graph import is_complete_tifxyz


def test_complete_tifxyz_is_recognized_and_a_bad_channel_is_not(tmp_path):
    grid = np.full((4, 6, 3), -1.0, dtype=np.float32)
    grid[1:3, 1:5] = 7.0
    out = tmp_path / 'band.tifxyz'
    tifxyz.write(out, tifxyz.Surface(grid, scale=(0.5, 0.25), extra={'seed_track_row': 3}),
                 compression='none', tile=None, mask=False)

    assert is_complete_tifxyz(out)
    meta = json.loads((out / 'meta.json').read_text())
    assert meta['scale'] == [0.25, 0.5] and meta['seed_track_row'] == 3
    assert meta['bbox'] == [[7.0, 7.0, 7.0], [7.0, 7.0, 7.0]]

    (out / 'y.tif').write_bytes(b'not a tiff at all' * 8)
    assert not is_complete_tifxyz(out)
    assert not is_complete_tifxyz(tmp_path / 'missing')
