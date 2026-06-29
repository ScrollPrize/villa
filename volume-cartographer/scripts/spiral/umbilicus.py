import json
import numpy as np
import scipy.interpolate


def read_thaumato_umbilicus(path):
    with open(path, 'rt') as fp:
        lines = fp.readlines()
    xyzs = np.asarray([
        tuple(map(int, line.strip().replace(' ', '').split(',')))
        for line in lines
    ])[:, [2, 0, 1]] - 500
    return xyzs[:, 2], xyzs[:, :2]


def thaumato_umbilicus_z_to_yx(filename, downsample_factor):
    zs, xys = read_thaumato_umbilicus(filename)
    zs = zs.astype(np.float32) / downsample_factor
    xys = xys.astype(np.float32)[:, ::-1] / downsample_factor
    return scipy.interpolate.interp1d(zs, xys, axis=0, fill_value='extrapolate')  # z -> xy


def json_umbilicus_z_to_yx(filename, downsample_factor):
    with open(filename, 'rt') as fp:
        data = json.load(fp)
    pts = sorted(data['control_points'], key=lambda p: p['z'])
    zs = np.asarray([p['z'] for p in pts], dtype=np.float32) / downsample_factor
    yxs = np.asarray([(p['y'], p['x']) for p in pts], dtype=np.float32) / downsample_factor
    return scipy.interpolate.interp1d(zs, yxs, axis=0, fill_value='extrapolate')  # z -> yx
