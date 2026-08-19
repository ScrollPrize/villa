"""Regression tests for the patch-strip loss's theta=0 unwrap on strips that
span multiple windings (whole-band patches such as 0000_top_band).

The patch losses sample P sorted picks along a strip (a grid-line run, a
dijkstra geodesic, or a serpentine 2D walk) and stitch theta=0 crossings from
consecutive-pick raw diffs. On a band whose strips span tens of windings the
picks sit more than pi apart in theta and the detector misassigns winding
offsets even on a perfectly-fit patch (the patch analogue of the long-fiber
bug). These tests pin that the cached source-topology crossings make a
geometrically perfect multi-wrap band read ~zero radius and DT loss in both
strip-sampling modes."""

import numpy as np
import pytest
import torch

from config import Config
import losses
from losses import get_unverified_patch_losses
from theta_crossing_map import ThetaCrossingMap

DR = 12.0
CELL = 20.0

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
        self.node_maps = []

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

    def register_topology(self, crossing_map):
        for patch_idx, grid in enumerate(self.grids):
            h, w = grid.shape[0] - 1, grid.shape[1] - 1
            centres = (
                grid[:-1, :-1] + grid[1:, :-1]
                + grid[:-1, 1:] + grid[1:, 1:]) / 4
            centres = centres.reshape(-1, 3).to(self.device)
            start = crossing_map.register_nodes(
                h * w,
                lambda indices, values=centres: values[indices])
            node_map = start + np.arange(h * w, dtype=np.int64).reshape(h, w)
            self.node_maps.append(node_map)
            edges = []
            for di, dj in ((0, 1), (1, -1), (1, 0), (1, 1)):
                a = node_map[:h - di, max(0, -dj):w - max(0, dj)]
                b = node_map[di:, max(0, dj):w - max(0, -dj)]
                edges.append(np.stack([a.reshape(-1), b.reshape(-1)], axis=1))
            crossing_map.register_edges(np.concatenate(edges))

    def theta_node_ids(self, patch_indices, ijs):
        patch_indices = np.asarray(patch_indices)
        cells = np.floor(ijs).astype(np.int64)
        out = np.empty(cells.shape[:-1], dtype=np.int64)
        for patch_idx in np.unique(patch_indices):
            which = patch_indices == patch_idx
            picked = cells[which]
            out[which] = self.node_maps[patch_idx][picked[:, 0], picked[:, 1]]
        return out


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


def _run_losses(grid, cfg, monkeypatch, num_steps=6, seed=0, device='cpu'):
    monkeypatch.setattr(
        losses.strip_path_pools, 'ensure_patch_path_pools', lambda patches: None)
    monkeypatch.setattr(
        losses.strip_path_pools, 'submit_patch_pool_refresh', lambda patch: None)
    np.random.seed(seed)
    torch.manual_seed(seed)
    patch = FakePatch(grid, cfg['patch_strip_sampling'])
    atlas = FakeAtlas([grid], device)
    crossing_map = ThetaCrossingMap(device)
    atlas.register_topology(crossing_map)
    crossing_map.force_refresh(IdentityTransform())
    dr = torch.tensor(DR, device=device)
    radius_losses, dt_losses = [], []
    for _ in range(num_steps):
        radius_loss, dt_loss = get_unverified_patch_losses(
            IdentityTransform(), dr, 1, 1, [patch], atlas,
            np.array([1.0]), compute_dt=True,
            crossing_map=crossing_map, cfg=cfg)
        radius_losses.append(float(radius_loss))
        dt_losses.append(float(dt_loss))
    return np.array(radius_losses), np.array(dt_losses)


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


@pytest.mark.parametrize('mode', ['straight', 'dijkstra'])
def test_crossing_map_handles_short_strips(mode, monkeypatch):
    cfg = Config().as_dict()
    cfg['patch_strip_sampling'] = mode
    grid = _band_grid(0.8, theta0=1.75 * np.pi)
    radius, dt = _run_losses(grid, cfg, monkeypatch, seed=3)
    assert radius.max() < 1e-3
    assert dt.max() < 1e-3
