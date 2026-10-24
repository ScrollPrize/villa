#!/usr/bin/env python3
"""Compare one pyramid level of two render zarrs voxel by voxel.

usage: cmp_zarr.py A.zarr B.zarr [LEVEL]   (LEVEL defaults to 0)
"""
import json
import os
import sys

import numpy as np
import zarr

lv = sys.argv[3] if len(sys.argv) > 3 else "0"
a, b = (zarr.open(os.path.join(p, lv), mode="r") for p in sys.argv[1:3])
assert a.shape == b.shape, (a.shape, b.shape)
n_diff = 0
max_diff = 0
for y in range(0, a.shape[1], 1024):
    d = np.abs(a[:, y:y + 1024, :].astype(np.int16) - b[:, y:y + 1024, :].astype(np.int16))
    n_diff += int(np.count_nonzero(d))
    max_diff = max(max_diff, int(d.max(initial=0)))
print(json.dumps(dict(level=lv, identical=n_diff == 0, n_diff=n_diff, max_diff=max_diff)))
