"""Regression tests for the patch-strip loss's theta=0 unwrap on strips that
span multiple windings (whole-band patches such as 0000_top_band).

The patch losses sample P sorted picks along a strip (a grid-line run, a
dijkstra geodesic, or a serpentine 2D walk) and stitch theta=0 crossings from
consecutive-pick raw diffs. On a band whose strips span tens of windings the
picks sit more than pi apart in theta and the detector misassigns winding
offsets even on a perfectly-fit patch (the patch analogue of the long-fiber
bug). The fix recomputes crossing adjustments along the strip's dense walk for
strips whose walk exceeds _PATCH_DENSE_UNWRAP_WALK_FACTOR * P cells; these
tests pin that a geometrically perfect multi-wrap band reads ~zero radius and
DT loss in both strip-sampling modes, that the dense path reproduces the
sparse unwrap exactly where the latter was already correct, and that the
straight-strip walk reconstruction is faithful."""

import numpy as np
import pytest
import torch

from config import Config
import losses
from losses import (
    PackedDenseWalks,
    _dense_walk_crossing_adjustments,
    _reconstruct_straight_dense_walks,
    get_unverified_patch_losses,
)

DR = 12.0
CELL = 20.0

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='_sample_patch_batch uploads the batch with .cuda()')


class IdentityTransform:
    def __call__(self, zyxs):
        return zyxs

    def inv(self, spiral_zyxs):
        return spiral_zyxs


def _band_grid(wraps, winding=66, num_rows=4, theta0=0.0):
    # (num_rows, W, 3) zyxs grid lying exactly on the spiral sheet
    # r = DR * (winding + theta / 2pi): columns march along the winding at
    # CELL arc spacing, rows step in z. Matches the tifxyz band patches'
    # geometry (rows follow the papyrus around the umbilicus).
    thetas = []
    theta = theta0
    while theta < theta0 + wraps * 2 * np.pi:
        thetas.append(theta)
        theta += CELL / (DR * (winding + theta / (2 * np.pi)))
    thetas = np.asarray(thetas)
    radii = DR * (winding + thetas / (2 * np.pi))
    row = np.stack([np.zeros_like(thetas), np.sin(thetas) * radii,
                    np.cos(thetas) * radii], axis=-1)
    grid = np.broadcast_to(row, (num_rows,) + row.shape).copy()
    grid[..., 0] = np.arange(num_rows, dtype=np.float64)[:, None] * CELL
    return torch.from_numpy(grid.astype(np.float32))


class FakeAtlas:
    # Minimal stand-in for fit_spiral.PatchAtlas: host-resident grids,
    # bilinear lookup, results moved to `device`. sampling_atlas None forces
    # the python sampling branch.
    sampling_atlas = None

    def __init__(self, grids, device):
        self.grids = grids
        self.device = torch.device(device)

    def lookup(self, patch_idx_per_sample, ijs):
        idx = patch_idx_per_sample.reshape(-1)
        pts = ijs.reshape(-1, 2)
        out = torch.empty(pts.shape[0], 3, dtype=torch.float32)
        for k in range(pts.shape[0]):
            g = self.grids[int(idx[k])]
            i = min(max(float(pts[k, 0]), 0.0), g.shape[0] - 1 - 1e-4)
            j = min(max(float(pts[k, 1]), 0.0), g.shape[1] - 1 - 1e-4)
            i0, j0 = int(i), int(j)
            di, dj = i - i0, j - j0
            out[k] = ((1 - di) * (1 - dj) * g[i0, j0]
                      + (1 - di) * dj * g[i0, j0 + 1]
                      + di * (1 - dj) * g[i0 + 1, j0]
                      + di * dj * g[i0 + 1, j0 + 1])
        return out.reshape(*ijs.shape[:-1], 3).to(device=self.device)


class FakePatch:
    def __init__(self, grid, mode):
        H, W = grid.shape[:2]
        self._sampling_2d_path = None
        if mode == 'straight':
            self._sampling_valid_quad_rows = np.arange(H - 1, dtype=np.int64)
            self._sampling_valid_quad_cols = np.arange(W - 1, dtype=np.int64)
            self._h_runs_los = [np.array([0])] * (H - 1)
            self._h_runs_his = [np.array([W - 1])] * (H - 1)
            self._h_runs_cum = [np.array([W - 1])] * (H - 1)
            self._v_runs_los = [np.array([0])] * (W - 1)
            self._v_runs_his = [np.array([H - 1])] * (W - 1)
            self._v_runs_cum = [np.array([H - 1])] * (W - 1)
        else:  # dijkstra: one geodesic along the band per pool slot
            path = np.stack([
                np.full(W - 1, 1, dtype=np.int64),
                np.arange(W - 1, dtype=np.int64),
            ], axis=1)
            self._strip_path_pool = [path]


def _run_losses(grid, cfg, monkeypatch, num_steps=6, seed=0, device='cuda'):
    monkeypatch.setattr(
        losses.strip_path_pools, 'ensure_patch_path_pools', lambda patches: None)
    monkeypatch.setattr(
        losses.strip_path_pools, 'submit_patch_pool_refresh', lambda patch: None)
    np.random.seed(seed)
    torch.manual_seed(seed)
    patch = FakePatch(grid, cfg['patch_strip_sampling'])
    atlas = FakeAtlas([grid], device)
    dr = torch.tensor(DR, device=device)
    radius_losses, dt_losses = [], []
    for _ in range(num_steps):
        radius_loss, dt_loss = get_unverified_patch_losses(
            IdentityTransform(), dr, 1, 1, [patch], atlas,
            np.array([1.0]), compute_dt=True, cfg=cfg)
        radius_losses.append(float(radius_loss))
        dt_losses.append(float(dt_loss))
    return np.array(radius_losses), np.array(dt_losses)


@requires_cuda
@pytest.mark.parametrize('mode', ['straight', 'dijkstra'])
def test_multiwrap_perfect_band_has_zero_loss(mode, monkeypatch):
    # A perfect 40-wrap band strip sampled at 400 picks: consecutive picks can
    # sit more than pi apart in theta, so the sparse unwrap this regresses
    # against misassigned most picks' winding offsets on every step.
    cfg = Config().as_dict()
    cfg['patch_strip_sampling'] = mode
    grid = _band_grid(40.0)
    radius_losses, dt_losses = _run_losses(grid, cfg, monkeypatch)
    assert radius_losses.max() < 1e-3
    assert dt_losses.max() < 1e-3


@requires_cuda
def test_legacy_sparse_unwrap_fails_on_multiwrap_band(monkeypatch):
    # Sanity check that the scenario above actually regresses something:
    # disabling the dense walk (factor so large it never fires) restores the
    # legacy sparse unwrap, which must misread the same perfect band.
    cfg = Config().as_dict()
    cfg['patch_strip_sampling'] = 'straight'
    monkeypatch.setattr(losses, '_PATCH_DENSE_UNWRAP_WALK_FACTOR', 10 ** 9)
    grid = _band_grid(40.0)
    radius_losses, _ = _run_losses(grid, cfg, monkeypatch)
    assert radius_losses.max() > 1e-3


@requires_cuda
@pytest.mark.parametrize('mode', ['straight', 'dijkstra'])
def test_dense_walk_matches_sparse_unwrap_on_short_strips(mode, monkeypatch):
    # On a sub-wrap band crossing the theta=0 seam the sparse unwrap is
    # already correct; forcing the dense walk onto every strip (factor 0) must
    # reproduce its losses exactly (same RNG, so the picks are identical).
    cfg = Config().as_dict()
    cfg['patch_strip_sampling'] = mode
    grid = _band_grid(0.8, theta0=1.75 * np.pi)
    monkeypatch.setattr(losses, '_PATCH_DENSE_UNWRAP_WALK_FACTOR', 10 ** 9)
    legacy_radius, legacy_dt = _run_losses(grid, cfg, monkeypatch, seed=3)
    monkeypatch.setattr(losses, '_PATCH_DENSE_UNWRAP_WALK_FACTOR', 0)
    dense_radius, dense_dt = _run_losses(grid, cfg, monkeypatch, seed=3)
    assert legacy_radius.max() < 1e-3
    np.testing.assert_array_equal(legacy_radius, dense_radius)
    np.testing.assert_array_equal(legacy_dt, dense_dt)


def test_reconstruct_straight_dense_walks():
    # Slot 0 fixes axis 0 (a row strip): the walk must cover exactly the
    # picks' floor-cell range at quad centres, with positions the picks'
    # offsets into it; a strip below the length threshold yields no entry.
    P = 4
    ijs = np.zeros([2, 1, P, 2], dtype=np.float32)
    ijs[0, 0, :, 0] = 7.3
    ijs[0, 0, :, 1] = [2.5, 30.1, 55.9, 90.2]
    ijs[1, 0, :, 1] = 3.7
    ijs[1, 0, :, 0] = [1.2, 2.5, 3.1, 4.9]  # span 4 cells <= 2 * P: skipped
    walks = _reconstruct_straight_dense_walks(ijs, P, skip=set())
    assert len(walks) == 1
    walk = walks[0]
    assert walk.row == 0
    assert walk.path.shape == (89, 2)
    np.testing.assert_allclose(walk.path[:, 0], 7.3, rtol=1e-6)
    np.testing.assert_allclose(walk.path[:, 1], np.arange(2, 91) + 0.5)
    np.testing.assert_array_equal(walk.pick_positions, [0, 28, 53, 88])


def test_dense_walk_connects_quad_centres_to_jittered_picks():
    # The shared dense-walk helper must include the final theta step from each
    # path cell's centre to the actual fractional patch pick. Flattened row 1
    # is selected here; row 0 is deliberately unrelated.
    walk_theta = torch.tensor([0.1, 0.2])
    walk_zyxs = torch.stack([
        torch.zeros_like(walk_theta),
        torch.sin(walk_theta),
        torch.cos(walk_theta),
    ], dim=-1)[None]
    sampled_theta = torch.tensor([
        [1.0, 1.1],
        [2 * np.pi - 0.1, 0.2],
    ])
    packed = PackedDenseWalks(
        rows=torch.tensor([1]),
        walk_zyxs=walk_zyxs,
        pick_positions=torch.tensor([[0, 1]]),
    )

    adjustments, rows = _dense_walk_crossing_adjustments(
        IdentityTransform(), torch.tensor(DR), sampled_theta, packed)

    assert torch.equal(rows, torch.tensor([1]))
    assert torch.allclose(adjustments, torch.tensor([[0.0, -DR]]))
