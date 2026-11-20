import json
import numpy as np
import scipy.interpolate


def _z_interp_hold_ends(zs, vals):
    # Past the first and last control points, hold the end point (as VC3D's Umbilicus::interpolate_center does)
    # instead of extending the last segment in a straight line, which can swing the centre by many mm.
    order = np.argsort(zs, kind='stable')
    zs, vals = zs[order], vals[order]
    return scipy.interpolate.interp1d(zs, vals, axis=0, bounds_error=False, fill_value=(vals[0], vals[-1]), assume_sorted=True)


def read_thaumato_umbilicus(path):
    with open(path, 'rt') as fp:
        lines = fp.readlines()
    xyzs = np.asarray([
        tuple(map(int, line.strip().replace(' ', '').split(',')))
        for line in lines
    ])[:, [2, 0, 1]] - 500
    return xyzs[:, 2], xyzs[:, :2]


def thaumato_umbilicus_z_to_yx(filename, coordinate_scale=1.0, downsample_factor=None):
    if downsample_factor is not None:
        coordinate_scale = 1.0 / downsample_factor
    zs, xys = read_thaumato_umbilicus(filename)
    zs = zs.astype(np.float32) * coordinate_scale
    xys = xys.astype(np.float32)[:, ::-1] * coordinate_scale
    return _z_interp_hold_ends(zs, xys)  # z -> xy


def json_umbilicus_z_to_yx(filename, coordinate_scale=1.0, downsample_factor=None):
    if downsample_factor is not None:
        coordinate_scale = 1.0 / downsample_factor
    with open(filename, 'rt') as fp:
        data = json.load(fp)
    pts = sorted(data['control_points'], key=lambda p: p['z'])
    zs = np.asarray([p['z'] for p in pts], dtype=np.float32) * coordinate_scale
    yxs = np.asarray([(p['y'], p['x']) for p in pts], dtype=np.float32) * coordinate_scale
    return _z_interp_hold_ends(zs, yxs)  # z -> yx
