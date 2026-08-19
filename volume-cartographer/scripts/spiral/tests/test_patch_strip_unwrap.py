"""Uniform 2D patch sampling remains exact on multi-wrap surfaces."""

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
            preorder = np.concatenate([
                node_map[row] if row % 2 == 0 else node_map[row, ::-1]
                for row in range(h)
            ])
            crossing_map.register_unwrap_tree(
                preorder, np.arange(-1, preorder.size - 1, dtype=np.int64))

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
    def __init__(self, grid):
        H, W = grid.shape[:2]
        self._sampling_valid_quad_mask_np = np.ones(
            (H - 1, W - 1), dtype=bool)
        self._sampling_valid_quad_indices_np = np.argwhere(
            self._sampling_valid_quad_mask_np).astype(np.int64)


def _run_losses(grid, cfg, num_steps=6, seed=0, device='cpu'):
    np.random.seed(seed)
    torch.manual_seed(seed)
    patch = FakePatch(grid)
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


def test_multiwrap_perfect_band_has_zero_loss():
    # A perfect 40-wrap band strip sampled at 400 picks: consecutive picks can
    # sit more than pi apart in theta, so the sparse unwrap this regresses
    # against misassigned most picks' winding offsets on every step.
    cfg = Config().as_dict()
    grid = _band_grid(40.0)
    radius_losses, dt_losses = _run_losses(grid, cfg)
    assert radius_losses.max() < 1e-3
    assert dt_losses.max() < 1e-3


def test_uniform_sampling_handles_short_ragged_patches():
    cfg = Config().as_dict()
    grid = _band_grid(0.8, theta0=1.75 * np.pi)
    radius, dt = _run_losses(grid, cfg, seed=3)
    assert radius.max() < 1e-3
    assert dt.max() < 1e-3
