import numpy as np
import pytest

import vc_tomography


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


@pytest.mark.parametrize("device", vc_tomography.devices())
def test_disk_phantom(device):
    n, n_angles, radius = 256, 720, 70.0
    c = (n - 1) / 2
    sino = ramp_filtered_disk_sinogram(n, n_angles, radius, c)
    angles = np.linspace(0, np.pi, n_angles, endpoint=False).astype(np.float32)
    img = vc_tomography.backproject(sino, angles, n, n, rot_center=c, scale=np.pi / n_angles, device=device)
    yy, xx = np.mgrid[:n, :n] - c
    r = np.hypot(xx, yy)
    inside, outside = img[r < radius - 5], img[(r > radius + 5) & (r < c - 5)]
    assert abs(inside.mean() - 1) < 2e-3 and inside.std() < 5e-3
    assert abs(outside.mean()) < 2e-3


@pytest.mark.skipif(not vc_tomography.cuda_available(), reason="no CUDA")
def test_cpu_cuda_agree_offcenter_roi_clip():
    rng = np.random.default_rng(1)
    sino = rng.standard_normal((333, 300)).astype(np.float32)
    angles = rng.uniform(-7, 7, 333).astype(np.float32)
    kw = dict(ny=90, nx=110, rot_center=171.3, scale=0.7, center=200.0, x0=40, y0=150,
              clip_outer_circle=True, clip_radius=120.0)
    a = vc_tomography.backproject(sino, angles, device="cpu", **kw)
    b = vc_tomography.backproject(sino, angles, device="cuda", **kw)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-3 * np.abs(a).max())
    assert (a == 0).any() and (a != 0).any()  # clip circle crosses the ROI


@pytest.mark.parametrize("device", vc_tomography.devices())
def test_only_the_listed_tiles_are_computed(device):
    rng = np.random.default_rng(2)
    sino = rng.standard_normal((200, 180)).astype(np.float32)
    angles = rng.uniform(-7, 7, 200).astype(np.float32)
    ny, nx, T = 70, 101, 32
    kw = dict(ny=ny, nx=nx, rot_center=88.2, scale=0.7, center=60.0, x0=5, y0=9, interp=1, device=device)
    full = vc_tomography.backproject(sino, angles, **kw)
    ntx, nty = -(-nx // T), -(-ny // T)
    for tiles in ([], [0], [5, 2, 11], list(range(ntx * nty))):     # tile ty * ntx + tx; the others are 0
        keep = np.zeros((ny, nx), bool)
        for t in tiles:
            keep[(t // ntx) * T:(t // ntx + 1) * T, (t % ntx) * T:(t % ntx + 1) * T] = True
        np.testing.assert_array_equal(vc_tomography.backproject(sino, angles, tiles=tiles, **kw), np.where(keep, full, 0))
    for bad in ([ntx * nty], [1, 1]):
        with pytest.raises(RuntimeError, match="tiles"):
            vc_tomography.backproject(sino, angles, tiles=bad, **kw)


# ---------------------------------------------------------------- a diverging beam (BackprojectParams::fan)

def blobs_sinogram(points, sigma, n, angles, rot, fan):
    """Line integrals of Gaussian blobs (unit peak, width sigma) along the rays of a beam that diverges: the ray
    recorded at the column m of the row at the backprojector's angle a is the parallel beam's (x cos a + y sin a =
    m - rot) turned about its point nearest the axis towards the angle a - fan[m]. fan 0: the parallel beam."""
    t = np.arange(n) - rot
    out = np.zeros((len(angles), n))
    for i, a in enumerate(angles):
        b = a - fan
        for px, py in points:
            d = (px - t * np.cos(a)) * np.cos(b) + (py - t * np.sin(a)) * np.sin(b)
            out[i] += np.exp(-d ** 2 / (2 * sigma ** 2)) * np.sqrt(2 * np.pi) * sigma
    return out


def ramp(sino):
    n = sino.shape[1]
    k = np.arange(-n + 1, n)
    h = np.zeros(k.size)
    h[k == 0] = 0.25
    odd = k % 2 != 0
    h[odd] = -1 / (np.pi * k[odd]) ** 2
    return np.array([np.convolve(row, h, mode="full")[n - 1: 2 * n - 1] for row in sino]).astype(np.float32)


FAN_N, FAN_ROT, FAN_ANGLES = 256, 127.5, 720
FAN_POINTS = [(0.0, 0.0), (70.0, 0.0), (0.0, -85.0), (-60.0, 55.0), (40.0, 80.0), (-75.0, -50.0), (95.0, -20.0)]


def fan_tables():
    m = np.arange(FAN_N)
    line = 1.5e-4 * (m - (FAN_ROT - 40))                    # one scan: the middle column 40 columns from the axis
    # two scans side by side, the second 150 columns on: each column's fan counts from its own scan's middle, and
    # over the 16 columns both hold it goes from one to the other
    first, second = 1.5e-4 * (m - 60.0), 1.5e-4 * (m - 210.0)
    share = np.clip((m - 142) / 16.0, 0, 1)
    return {"one scan": line, "two scans": (1 - share) * first + share * second}


@pytest.mark.parametrize("device", vc_tomography.devices())
@pytest.mark.parametrize("which", ["one scan", "two scans"])
def test_a_diverging_beam_is_backprojected_along_its_rays(device, which):
    """Blobs recorded by a diverging beam, filtered as a parallel beam's rows and backprojected with the beam's fan,
    come out where and as they are (as a parallel beam's sinogram backprojected without one does); backprojected as
    a parallel beam's they are turned and smeared by the fan x their depth (here up to 2 px)."""
    fan = fan_tables()[which].astype(np.float32)
    angles = (np.arange(FAN_ANGLES) * (2 * np.pi / FAN_ANGLES)).astype(np.float32)
    kw = dict(ny=FAN_N, nx=FAN_N, rot_center=FAN_ROT, scale=np.pi / FAN_ANGLES, interp=1, device=device)
    parallel = vc_tomography.backproject(ramp(blobs_sinogram(FAN_POINTS, 2.5, FAN_N, angles, FAN_ROT, 0.0)), angles, **kw)
    recorded = ramp(blobs_sinogram(FAN_POINTS, 2.5, FAN_N, angles, FAN_ROT, fan))
    with_fan = vc_tomography.backproject(recorded, angles, fan=list(fan), **kw)
    without = vc_tomography.backproject(recorded, angles, **kw)
    assert 0.93 < parallel.max() < 1.03
    yy, xx = np.mgrid[:FAN_N, :FAN_N] - FAN_ROT
    inside = np.hypot(xx, yy) < 110
    left, as_parallel = np.abs(with_fan - parallel)[inside].max(), np.abs(without - parallel)[inside].max()
    assert left < 0.012 and as_parallel > 0.05 and as_parallel > 5 * left, (left, as_parallel)
    # the fan in the other sense doubles what a parallel beam leaves
    other = vc_tomography.backproject(recorded, angles, fan=list(-fan), **kw)
    assert np.abs(other - parallel)[inside].max() > 1.7 * as_parallel
    # a fan of nothing is the parallel beam's backprojection
    np.testing.assert_allclose(vc_tomography.backproject(recorded, angles, fan=[0.0] * FAN_N, **kw)[inside], without[inside], atol=2e-6)


@pytest.mark.skipif(not vc_tomography.cuda_available(), reason="no CUDA")
def test_cpu_cuda_agree_with_a_fan():
    rng = np.random.default_rng(3)
    sino = rng.standard_normal((333, 300)).astype(np.float32)
    angles = rng.uniform(-7, 7, 333).astype(np.float32)
    fan = list((2e-4 * (np.arange(300) - 120.0)).astype(np.float32))
    for interp in (0, 3):
        kw = dict(ny=90, nx=110, rot_center=171.3, scale=0.7, center=200.0, x0=40, y0=150, clip_outer_circle=True, clip_radius=120.0, interp=interp, fan=fan)
        a = vc_tomography.backproject(sino, angles, device="cpu", **kw)
        b = vc_tomography.backproject(sino, angles, device="cuda", **kw)
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=2e-3 * np.abs(a).max())
    with pytest.raises(RuntimeError, match="fan"):
        vc_tomography.backproject(sino, angles, ny=8, nx=8, rot_center=150.0, fan=[])
