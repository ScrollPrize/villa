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


# What the model transform smoothed its umbilicus with before the width was a setting: checkpoints and
# surfaces written then were fitted around the axis smoothed this much.
LEGACY_UMBILICUS_SMOOTHING_SIGMA = 75.0


def check_umbilicus_smoothing_sigma(sigma):
    """``sigma`` as a float, or ValueError: the width, in z slices of the fit, is a finite number >= 0 (0: none)."""
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not np.isfinite(sigma) or sigma < 0:
        raise ValueError(f'umbilicus_smoothing_sigma must be a finite number >= 0, got {sigma!r}')
    return float(sigma)


def smooth_umbilicus_z_to_yx(z_to_yx, z_begin, z_end, sigma):
    """The umbilicus curve every consumer of a fit reads: the model transform, the umbilicus loss, the shell
    polar table, the metrics and the checkpoint readers.

    The curve is sampled at the integer slices z_begin..z_end-1 of the fit, smoothed with a Gaussian of
    ``sigma`` slices (0: not smoothed), and read back as the model transform reads its table: linearly between
    slices, the first and last segments extended beyond the ends. Sampling first is what keeps a fractional
    slice, and a slice in the margin around the fit, on the transform's side of the table.
    """
    sigma = check_umbilicus_smoothing_sigma(sigma)
    zs = np.arange(z_begin, z_end)
    table = np.asarray(z_to_yx(zs), dtype=np.float32)
    if sigma > 0:
        table = scipy.ndimage.gaussian_filter1d(table, sigma=sigma, axis=0, mode='nearest')
    table = table.astype(np.float64)
    if len(zs) < 2:
        return lambda z: np.broadcast_to(table[0], np.shape(z) + (2,)).copy()

    def curve(z):
        z = np.asarray(z, dtype=np.float64)
        # torch.searchsorted(zs, z) - 1, clamped to the segments: what geom_utils.interp1d does
        at = np.clip(np.searchsorted(zs, z, side='left') - 1, 0, len(zs) - 2)
        return table[at] + (table[at + 1] - table[at]) * (z - zs[at])[..., None]
    return curve


def umbilicus_zyx_table(z_to_yx, z_begin, z_end, sigma):
    """The float32 (z, y, x) table of the smoothed curve at the integer slices z_begin..z_end-1, which is what the
    model transform (``UmbilicusTransform``) is built from."""
    zs = np.arange(z_begin, z_end)
    centre = smooth_umbilicus_z_to_yx(z_to_yx, z_begin, z_end, sigma)(zs)
    return np.concatenate([zs[:, None], centre], axis=-1).astype(np.float32)
