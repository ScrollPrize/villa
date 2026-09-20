"""The interactive preview surface is spliced onto satisfied patches.

The final ``_spliced`` meshes snap the transformed surface onto every patch
that passes the splicing satisfaction profile. The live preview does the same
so what VC3D shows during a session is what the final export will sit on.
"""
import math
from types import SimpleNamespace
from unittest import mock

import torch

import spiral_helpers
from sample_spiral import get_spiral_yxs
from satisfaction_metrics import (
    _ListPatchAtlas,
    evaluate_patch_satisfaction_packed,
)

DR = 10.0
STEP = 2


class IdentityTransform:
    def __call__(self, points):
        return points

    def inv(self, points):
        return points


def spiral_vertex(theta, winding, z):
    radius = (winding + theta / (2 * math.pi)) * DR
    return [z, math.sin(theta) * radius, math.cos(theta) * radius]


def patch_on_spiral(winding, thetas, zs):
    """A patch lying on the spiral at a (possibly fractional) winding."""
    zyxs = torch.tensor(
        [[spiral_vertex(theta, winding, z) for theta in thetas] for z in zs],
        dtype=torch.float32)
    valid = torch.ones((len(zs) - 1, len(thetas) - 1), dtype=torch.bool)
    return SimpleNamespace(zyxs=zyxs, valid_quad_mask=valid,
                           area=float(valid.sum()))


def export(patches, evaluation, atlas):
    cfg = {"shell_outer_winding_idx": 12, "output_step_size": STEP,
           "model_flow_bounds_z_margin": 0}
    written = {}

    def capture(winding_grids, *args, **kwargs):
        written.update(winding_grids)
        return {"windings": sorted(winding_grids)}

    with mock.patch.object(spiral_helpers, "save_combined_tifxyz",
                           side_effect=capture), \
         mock.patch.object(
             spiral_helpers, "get_spiral_yxs",
             side_effect=lambda *args, **kwargs: get_spiral_yxs(
                 *args, **kwargs, device="cpu")):
        spiral_helpers.save_combined_preview(
            IdentityTransform(), torch.tensor(DR), patches, [],
            "/unused", cfg, z_begin=0, z_end=40, voxel_size_um=9.6,
            get_or_build_unattached_pcl_flat=lambda *_: None,
            surface_id="surface", patch_atlas=atlas,
            patch_satisfaction_evaluation=evaluation)
    return written


def radii(grid):
    valid = ~(grid == -1.0).all(axis=-1)
    return torch.as_tensor(grid[valid][:, 1:]).norm(dim=-1), valid


def test_preview_surface_is_spliced_onto_satisfied_patches():
    # Off the spiral by 0.3 dr (inside the splicing tolerance), so where the
    # surface lands tells the splice apart from the model.
    thetas = [0.1 * step for step in range(1, 12)]
    zs = [4.0 * step for step in range(10)]
    patch = patch_on_spiral(11.3, thetas, zs)
    patches = [patch]
    atlas = _ListPatchAtlas(patches, torch.device("cpu"))
    evaluation = evaluate_patch_satisfaction_packed(
        IdentityTransform(), torch.tensor(DR), patches, atlas, 0, 40,
        include_splicing=True)
    assert evaluation.profiles["splicing"].satisfied_patches.item()

    plain = export(patches, None, atlas)
    spliced = export(patches, evaluation, atlas)
    assert sorted(plain) == sorted(spliced) == [10, 11, 12]

    # Windings the patch does not cover are exactly the model surface.
    for winding in (10, 12):
        assert (plain[winding] == spliced[winding]).all()

    changed = (plain[11] != spliced[11]).any(axis=-1)
    assert changed.any()
    # Spliced vertices carry the patch's radius (11.3 windings out), not the
    # model's (11.0), and only under the patch's theta and z footprint.
    spliced_radii, _ = radii(spliced[11][changed])
    plain_radii, _ = radii(plain[11][changed])
    spliced_windings = spliced_radii / DR
    plain_windings = plain_radii / DR
    assert (spliced_windings > plain_windings + 0.2).all()
    assert (spliced_windings < 11.3 + thetas[-1] / (2 * math.pi) + 0.05).all()
    rows, columns = changed.nonzero()
    assert rows.min() >= 0 and rows.max() * STEP <= zs[-1]
    unchanged_thetas = torch.atan2(
        torch.as_tensor(plain[11][0, :, 1]), torch.as_tensor(plain[11][0, :, 2]))
    covered = unchanged_thetas[torch.as_tensor(columns).unique()]
    assert covered.min() >= thetas[0] - 0.05
    assert covered.max() <= thetas[-1] + 0.05


def test_splice_overlay_honours_the_preview_first_winding():
    """A grid that starts at winding 10 indexes its columns from winding 10."""
    thetas = [0.1 * step for step in range(1, 12)]
    zs = [4.0 * step for step in range(10)]
    patches = [patch_on_spiral(10.2, thetas, zs), patch_on_spiral(3.2, thetas, zs)]
    atlas = _ListPatchAtlas(patches, torch.device("cpu"))
    evaluation = evaluate_patch_satisfaction_packed(
        IdentityTransform(), torch.tensor(DR), patches, atlas, 0, 40,
        include_splicing=True)
    assert evaluation.profiles["splicing"].satisfied_patches.all()

    yxs_by_winding = get_spiral_yxs(13, torch.tensor(DR), STEP,
                                    group_by_winding=True, device="cpu")
    spiral_zs = torch.arange(0, 40, STEP, dtype=torch.float32)
    grids = []
    for winding in range(10, 13):
        yxs = yxs_by_winding[winding]
        grids.append(torch.cat([
            spiral_zs[:, None, None].expand(-1, yxs.shape[0], 1),
            yxs[None].expand(spiral_zs.shape[0], -1, 2)], dim=-1))
    num_thetas = [grid.shape[1] for grid in grids]
    combined = torch.cat(grids, dim=1)
    before = combined.clone()
    spiral_helpers._build_spliced_overlay(
        combined, num_thetas, 0, STEP, IdentityTransform(), torch.tensor(DR),
        atlas, evaluation, first_winding=10)
    changed = (combined != before).any(dim=-1)
    # Only winding 10's columns change: the winding-3 patch is below the grid
    # and must not be written into anyone else's columns.
    assert changed[:, :num_thetas[0]].any()
    assert not changed[:, num_thetas[0]:].any()
