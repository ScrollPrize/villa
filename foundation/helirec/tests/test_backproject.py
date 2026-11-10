import numpy as np
import pytest

import helirec


def ramp_filtered_disk_sinogram(n, n_angles, radius, center):
    """Ram-Lak filtered sinogram of a centered unit disk (exact Radon transform)."""
    t = np.arange(n) - center
    row = 2 * np.sqrt(np.clip(radius**2 - t**2, 0, None))
    k = np.arange(-n + 1, n)
    h = np.zeros(k.size)
    h[k == 0] = 0.25
    odd = k % 2 != 0
    h[odd] = -1 / (np.pi * k[odd]) ** 2
    filtered = np.convolve(row, h, mode="full")[n - 1 : 2 * n - 1]
    return np.tile(filtered, (n_angles, 1)).astype(np.float32)


@pytest.mark.parametrize("device", helirec.devices())
def test_disk_phantom(device):
    n, n_angles, radius = 256, 720, 70.0
    c = (n - 1) / 2
    sino = ramp_filtered_disk_sinogram(n, n_angles, radius, c)
    angles = np.linspace(0, np.pi, n_angles, endpoint=False).astype(np.float32)
    img = helirec.backproject(sino, angles, n, n, rot_center=c, scale=np.pi / n_angles, device=device)
    yy, xx = np.mgrid[:n, :n] - c
    r = np.hypot(xx, yy)
    inside, outside = img[r < radius - 5], img[(r > radius + 5) & (r < c - 5)]
    assert abs(inside.mean() - 1) < 2e-3 and inside.std() < 5e-3
    assert abs(outside.mean()) < 2e-3


@pytest.mark.skipif(not helirec.cuda_available(), reason="no CUDA")
def test_cpu_cuda_agree_offcenter_roi_clip():
    rng = np.random.default_rng(1)
    sino = rng.standard_normal((333, 300)).astype(np.float32)
    angles = rng.uniform(-7, 7, 333).astype(np.float32)
    kw = dict(ny=90, nx=110, rot_center=171.3, scale=0.7, center=200.0, x0=40, y0=150,
              clip_outer_circle=True, clip_radius=120.0)
    a = helirec.backproject(sino, angles, device="cpu", **kw)
    b = helirec.backproject(sino, angles, device="cuda", **kw)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-3 * np.abs(a).max())
    assert (a == 0).any() and (a != 0).any()  # clip circle crosses the ROI


@pytest.mark.parametrize("device", helirec.devices())
def test_only_the_listed_tiles_are_computed(device):
    rng = np.random.default_rng(2)
    sino = rng.standard_normal((200, 180)).astype(np.float32)
    angles = rng.uniform(-7, 7, 200).astype(np.float32)
    ny, nx, T = 70, 101, 32
    kw = dict(ny=ny, nx=nx, rot_center=88.2, scale=0.7, center=60.0, x0=5, y0=9, interp=1, device=device)
    full = helirec.backproject(sino, angles, **kw)
    ntx, nty = -(-nx // T), -(-ny // T)
    for tiles in ([], [0], [5, 2, 11], list(range(ntx * nty))):     # tile ty * ntx + tx; the others are 0
        keep = np.zeros((ny, nx), bool)
        for t in tiles:
            keep[(t // ntx) * T:(t // ntx + 1) * T, (t % ntx) * T:(t % ntx + 1) * T] = True
        np.testing.assert_array_equal(helirec.backproject(sino, angles, tiles=tiles, **kw), np.where(keep, full, 0))
    for bad in ([ntx * nty], [1, 1]):
        with pytest.raises(RuntimeError, match="tiles"):
            helirec.backproject(sino, angles, tiles=bad, **kw)
