import json
import numpy as np
import scipy.interpolate
import scipy.ndimage


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
    return scipy.interpolate.interp1d(zs, xys, axis=0, fill_value='extrapolate')  # z -> xy


def json_umbilicus_z_to_yx(filename, coordinate_scale=1.0, downsample_factor=None):
    if downsample_factor is not None:
        coordinate_scale = 1.0 / downsample_factor
    with open(filename, 'rt') as fp:
        data = json.load(fp)
    pts = sorted(data['control_points'], key=lambda p: p['z'])
    zs = np.asarray([p['z'] for p in pts], dtype=np.float32) * coordinate_scale
    yxs = np.asarray([(p['y'], p['x']) for p in pts], dtype=np.float32) * coordinate_scale
    return scipy.interpolate.interp1d(zs, yxs, axis=0, fill_value='extrapolate')  # z -> yx


def smooth_umbilicus_z_to_yx(z_to_yx, z_begin, z_end, sigma):
    """The umbilicus curve smoothed once (Gaussian, ``sigma`` slices) over the fit slab.

    The curve is sampled at the integer slices z_begin..z_end-1, smoothed, and
    read back by linear interpolation (held at the ends outside the slab). The
    model transform, the umbilicus loss, the shell polar table and the metrics
    all read this one curve. ``sigma`` <= 0 returns ``z_to_yx`` unchanged.
    """
    if sigma <= 0:
        return z_to_yx
    zs = np.arange(z_begin, z_end)
    yx = scipy.ndimage.gaussian_filter1d(
        np.asarray(z_to_yx(zs), dtype=np.float32), sigma=float(sigma), axis=0, mode='nearest')
    return lambda z: np.stack([np.interp(z, zs, yx[:, 0]), np.interp(z, zs, yx[:, 1])], axis=-1)
