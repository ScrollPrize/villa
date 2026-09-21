"""Pinned winding radii, transform level.

A synthetic Archimedean scroll (umbilicus on the z axis, spacing DR) with
patches, cross-patch PCLs (one crossing the theta = 0 seam, one absolute) and
an unattached strip; the real PatchAtlas / ThetaCrossingMap / SpiralAndTransform
machinery on the CPU. Covers registry offsets and the consistency report,
transform-level exactness (positive and negative cases), checkpoint round trips of the registry, and the pinned DT
target values.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pins
from fit_spiral import PatchAtlas
from sample_spiral import get_theta_and_radii
from spiral_helpers import SequenceChain
from theta_crossing_map import ThetaCrossingMap
from tifxyz import Patch
from transforms import (
    PinnedGapExpandingTransform, SpiralAndTransform, pinned_gap_stage,
)

DR = 16.0
TWO_PI = 2 * math.pi

TINY_CONFIG = {
    'model_flow_voxel_resolution': 16,
    'model_flow_field_type': 'cartesian',
    'model_num_flow_stages': 1,
    'model_flow_field_direct_lr': False,
    'model_linear_z_resolution': 48,
    'model_gap_expander_logit_resolution': 24,
    'model_gap_expander_num_windings': 12,
    'model_gap_expander_capacity_windings': 12,
    'model_initial_dr_per_winding': DR,
    'model_gap_expander_min_gap': 1.0,
    'model_gap_expander_softplus_bias': 4.0,
    'model_gap_expander_lr_scale': 0.3,
    'output_first_winding': 1,
    'model_pin_coincidence_frac': 0.05,
    'model_pin_conflict_tolerance': 0.1,
}


def make_model(seed=0, flow_std=0.0, gap_std=0.0):
    torch.manual_seed(seed)
    umbilicus = torch.zeros([5, 3])
    umbilicus[:, 0] = torch.linspace(0., 192., 5)
    model = SpiralAndTransform(
        flow_integration_steps=3, flow_integration_solver='rk4',
        flow_min_corner_zyx=torch.tensor([0, -192, -192]),
        flow_max_corner_zyx=torch.tensor([192, 192, 192]),
        umbilicus_zyx=umbilicus, config=dict(TINY_CONFIG))
    with torch.no_grad():
        if flow_std > 0:
            for flow in model.flow_field.flows:
                flow.normal_(std=flow_std)
        if gap_std > 0:
            model.gap_expander_params.logits.normal_(std=gap_std)
    return model


def spiral_point(winding, theta, z):
    """Scroll-space point on the ideal spiral (umbilicus at y = x = 0)."""
    r = DR * (winding + theta / TWO_PI)
    return np.array([z, r * math.sin(theta), r * math.cos(theta)], dtype=np.float32)


def make_patch(winding, theta0, theta1, z0, z1, n_theta=9, n_z=7):
    """A patch grid on one sheet; theta may run past 2pi (the seam)."""
    thetas = np.linspace(theta0, theta1, n_theta)
    zs = np.linspace(z0, z1, n_z)
    grid = np.zeros([n_z, n_theta, 3], dtype=np.float32)
    for i, z in enumerate(zs):
        for j, theta in enumerate(thetas):
            grid[i, j] = spiral_point(winding, theta % TWO_PI, z) if theta < TWO_PI else spiral_point(winding + 1, theta - TWO_PI, z)
    patch = Patch(zyxs=torch.from_numpy(grid), scale=torch.tensor([1.0, 1.0]),
                  overlapping_ids=None, winding=None, uuid=f'w{winding}')
    patch._sampling_valid_quad_mask_np = np.asarray(patch.valid_quad_mask.numpy(), dtype=bool)
    return patch


def make_pcl(pid, entries, absolute=False):
    """entries: list of (zyx, winding_annotation, on_patch or None)."""
    points = {}
    for key, (zyx, winding, on_patch) in enumerate(entries):
        point = {'zyx': np.asarray(zyx, dtype=np.float32), 'winding_annotation': float(winding)}
        if on_patch is not None:
            point['on_patch'] = on_patch
        points[key] = point
    pcl = {'id': pid, 'name': pid, 'points': points, 'metadata': {'winding_is_absolute': absolute}}
    pcl['chain'] = SequenceChain(pcl)
    pcl['points_by_patch'] = {}
    for point in points.values():
        if 'on_patch' in point:
            pcl['points_by_patch'].setdefault(point['on_patch']['id'], []).append(point)
    return pcl


def build_scene(*, inconsistent=False):
    """Two patches on adjacent sheets (raw windings 3 and 5 at their own
    angles, patch A straddling the seam so its far side is raw winding 4)
    linked by PCLs; an absolute PCL on patch B; an unattached strip on the
    sheet between them crossing the seam."""
    patches = {
        'A': make_patch(3, TWO_PI - 0.8, TWO_PI + 0.8, 40.0, 100.0),   # crosses theta = 0
        'B': make_patch(5, 1.0, 2.6, 60.0, 130.0),
    }
    atlas = PatchAtlas(patches, device='cpu')
    # PCL: A at (i=3, j=2) -> theta = 2pi - 0.4, raw winding 3; a middle point
    # on A's own sheet just past the seam (raw winding 4, theta 0.3); B at
    # (i=2, j=4) -> theta 1.8, raw winding 5. B's sheet is the one adjacent
    # to A's, so the annotations (raw winding differences along the chain,
    # seam crossings handled by the chain walk) are 0, 0, 1: the raw
    # difference A -> mid is +1 but entirely due to the seam.
    a_zyx = patches['A'].zyxs[3, 2].numpy()
    b_zyx = patches['B'].zyxs[2, 4].numpy()
    link = make_pcl('link', [
        (a_zyx, 0.0, {'id': 'A', 'ij': (3.2, 2.3)}),
        (spiral_point(4, 0.3, 75.0), 0.0, None),
        (b_zyx, 2.0 if inconsistent else 1.0, {'id': 'B', 'ij': (2.4, 4.1)}),
    ])
    absolute = make_pcl('abs', [
        (patches['B'].zyxs[4, 6].numpy(), 5.0, {'id': 'B', 'ij': (4.2, 6.1)}),
    ], absolute=True)
    # A second relative PCL closing a cycle (consistent or not).
    link2 = make_pcl('link2', [
        (patches['A'].zyxs[1, 6].numpy(), 0.0, {'id': 'A', 'ij': (1.5, 6.5)}),
        (patches['B'].zyxs[5, 1].numpy(), 1.0, {'id': 'B', 'ij': (5.5, 1.5)}),
    ])
    strip_thetas = np.linspace(TWO_PI - 0.6, TWO_PI + 0.6, 7)
    strip_zyxs = np.stack([
        spiral_point(4, t % TWO_PI, 90.0) if t < TWO_PI else spiral_point(5, t - TWO_PI, 90.0)
        for t in strip_thetas])
    strips = [{'id': 'strip', 'zyxs': strip_zyxs, 'windings': np.zeros(7, dtype=np.float32),
               'radial_offsets': None, 'link_points': {}}]
    return patches, atlas, [link, absolute, link2], strips


def build_registry(model, patches, atlas, pcls, strips, stride=1, z_range=None):
    graph = pins.build_pin_graph(
        verified_patches=patches, patch_atlas=atlas, cross_patch_pcls=pcls,
        unattached_pcl_strips=strips, unattached_components=[[0]],
        unattached_component_edges=[[]], patch_grid_stride=stride, z_range=z_range)
    crossing_map = ThetaCrossingMap('cpu')
    atlas.register_theta_topology(crossing_map)
    transform = model.get_slice_to_spiral_transform()
    crossing_map.force_refresh(transform)
    gap, _, _, _ = model._get_transform_parts(with_pins=False)

    def free_gap(theta, z, slot):
        table = gap.get_transformed_winding_radii(theta, z)
        gaps = table.diff(dim=-1)
        return torch.gather(gaps, -1, slot.clamp(0, gaps.shape[-1] - 1)[:, None]).squeeze(-1)

    registry = pins.finalize_registry(
        graph, intermediate_transform=model.get_slice_to_intermediate_transform(),
        dr=model.get_dr_per_winding(), crossing_map=crossing_map, patch_atlas=atlas,
        footprint_rule=pins.FootprintRule(), free_gap_fn=free_gap,
        min_z=0.0, max_z=192.0, device=torch.device('cpu'),
        canonical_transform=model.get_unpinned_slice_to_spiral_transform())
    return graph, registry


# ---------------------------------------------------------------------------
# registry offsets
# ---------------------------------------------------------------------------


def test_registry_offsets():
    model = make_model()   # identity: the scroll is the ideal spiral
    patches, atlas, pcls, strips = build_scene()
    graph, registry = build_registry(model, patches, atlas, pcls, strips)
    # A, B and both relative PCLs form one component; the strip another.
    assert graph.num_components == 2
    assert registry.num_pins == 8 * 6 * 2 + (3 + 1 + 2) + 7   # quad centres + pcl points + strip points
    assert registry.consistency_report['inconsistent_edges'] == 0
    # Under the identity model every pin's shifted/dr - n_i equals its
    # component's T (quad centres sit ~0.02 windings inside the exact spiral:
    # chord versus arc).
    with torch.no_grad():
        inter = model.get_slice_to_intermediate_transform()(registry.zyx)
        theta, _, shifted = get_theta_and_radii(inter[:, 1:], model.get_dr_per_winding())
    estimate = shifted / DR - registry.n0.float()
    for comp in range(registry.num_components):
        values = estimate[registry.component == comp]
        assert float(values.max() - values.min()) < 0.05, values
    # The patch component is absolute (T fixed by the abs PCL): the sheet's
    # true winding is recovered exactly.
    comp_patch = int(registry.component[registry.kind == pins.PIN_KIND_PATCH][0])
    assert bool(registry.fixed_T[comp_patch])
    T = registry.initial_T
    assert abs(float(T[comp_patch] + registry.patch_offset[atlas.id_to_idx['A']]) - 3.0) < 1e-3
    assert abs(float(T[comp_patch] + registry.patch_offset[atlas.id_to_idx['B']]) - 5.0) < 1e-3
    assert int(registry.patch_offset[atlas.id_to_idx['B']] - registry.patch_offset[atlas.id_to_idx['A']]) == 2
    # Hand values: pins on patch A on the far side of the seam carry n one
    # lower than those before it.
    a_mask = (registry.kind == pins.PIN_KIND_PATCH) & (registry.zyx[:, 0] <= 100.0) & (registry.zyx[:, 0] >= 40.0) \
        & (torch.linalg.norm(registry.zyx[:, 1:], dim=-1) < DR * 4.5)
    theta_a = registry.theta0[a_mask]
    n_a = registry.n0[a_mask]
    before_seam = theta_a > math.pi
    assert int(n_a[before_seam].unique().numel()) == 1 and int(n_a[~before_seam].unique().numel()) == 1
    assert int(n_a[~before_seam][0]) == int(n_a[before_seam][0]) + 1
    # The strip crosses the seam too: n steps by one across it, T ~ 4.
    strip_mask = registry.component != comp_patch
    assert abs(float(T[~registry.fixed_T][0]) - 4.0) < 1e-2
    assert set(registry.n0[strip_mask].tolist()) == {0, 1}

    # Inconsistent cycle: the second PCL disagrees with the first by one winding.
    patches, atlas, pcls, strips = build_scene(inconsistent=True)
    graph, registry = build_registry(model, patches, atlas, pcls, strips)
    report = registry.consistency_report
    assert report['inconsistent_edges'] >= 1
    assert any('link' in entry['edge'] or 'patch' in entry['edge']
               for entries in report['components'].values() for entry in entries)


# ---------------------------------------------------------------------------
# Transform level: exactness through the full chain, positive and negative
# ---------------------------------------------------------------------------


def _attach(model, registry, active=True):
    model.set_pin_registry(registry)
    model.pins_active = active


def test_pinned_transform_exact_on_registry_points():
    # The scroll is the ideal spiral; the model's flow is a random smooth
    # deformation with the ordering right but positions off, and the gap
    # logits are perturbed. The pinned transform must still place every
    # registry point exactly on its target winding.
    model = make_model(seed=3, flow_std=2e-3, gap_std=0.02)
    patches, atlas, pcls, strips = build_scene()
    graph, registry = build_registry(model, patches, atlas, pcls, strips)
    _attach(model, registry)
    transform = model.get_slice_to_spiral_transform()
    gap = pinned_gap_stage(transform)
    assert isinstance(gap, PinnedGapExpandingTransform)
    assert gap.pin_table.num_registry_pins == registry.num_pins
    with torch.no_grad():
        spiral = transform(registry.zyx)
        dr = model.get_dr_per_winding()
        theta, _, shifted = get_theta_and_radii(spiral[:, 1:], dr)
        n = registry.adjusted_n(theta)
        T = model.effective_pin_targets()
        target_shifted = dr * (T[registry.component] + n.float())
        diagnostics = gap.pin_diagnostics(model.last_pins)
    assert diagnostics['pin_order_violations'] == 0
    assert diagnostics['pin_min_rise_violations'] == 0
    assert len(model.pin_conflicts) == 0
    # The PCL point on A's sheet coincides with a quad centre: the pair is
    # merged (compatible) and pinned at its mean; every other pin is exact.
    merged = model.merged_pin_mask()
    assert int(merged.sum()) == 2
    residual = (shifted - target_shifted).abs()
    assert residual[~merged].max() < 2e-3, diagnostics
    assert residual[merged].max() < 0.1 * DR
    # Inverse round trip through the pinned chain.
    with torch.no_grad():
        back = transform.inv(spiral)
    assert (back - registry.zyx).abs().max() < 1e-2
    # The unpinned transform does not have this property (the pins do work).
    model.pins_active = False
    with torch.no_grad():
        free_spiral = model.get_slice_to_spiral_transform()(registry.zyx)
        _, _, free_shifted = get_theta_and_radii(free_spiral[:, 1:], dr)
    assert (free_shifted - target_shifted).abs().max() > 0.5


def test_pins_inactive_is_the_unpinned_transform():
    model = make_model(seed=4, flow_std=1e-3, gap_std=0.02)
    patches, atlas, pcls, strips = build_scene()
    graph, registry = build_registry(model, patches, atlas, pcls, strips)
    points = torch.stack([
        torch.empty([200]).uniform_(20., 170.),
        torch.empty([200]).uniform_(-120., 120.),
        torch.empty([200]).uniform_(-120., 120.),
    ], dim=-1)
    with torch.no_grad():
        before = model.get_slice_to_spiral_transform()(points)
        _attach(model, registry, active=False)
        after = model.get_slice_to_spiral_transform()(points)
    assert torch.equal(before, after)
    assert pinned_gap_stage(model.get_slice_to_spiral_transform()) is None


def test_ordering_guard_fires_where_windings_swap():
    # Negative case: the flow has swapped two sheets along some rays. Emulate
    # it in the data: patch B is placed in A's theta/z range but 2.2 windings
    # inside its annotated position, so on rays through both patches the
    # anchors sorted by radius have decreasing targets.
    model = make_model(seed=5)
    patches, atlas, pcls, strips = build_scene()
    b = make_patch(5, TWO_PI - 0.6, TWO_PI + 0.4, 60.0, 130.0)
    yx = b.zyxs[..., 1:]
    radius = torch.linalg.norm(yx, dim=-1, keepdim=True)
    patches['B'] = Patch(zyxs=torch.cat([b.zyxs[..., :1], yx * (radius - 2.2 * DR) / radius], dim=-1),
                         scale=b.scale, overlapping_ids=None, winding=None, uuid='w5')
    patches['B']._sampling_valid_quad_mask_np = np.asarray(patches['B'].valid_quad_mask.numpy(), dtype=bool)
    atlas = PatchAtlas(patches, device='cpu')
    # The PCLs keep claiming B is the sheet outside A (and absolutely winding 5).
    for pcl in pcls:
        for point in pcl['points'].values():
            if point.get('on_patch', {}).get('id') == 'B':
                ij = point['on_patch']['ij']
                point['zyx'] = patches['B'].zyxs[int(ij[0]), int(ij[1])].numpy()
    graph, registry = build_registry(model, patches, atlas, pcls, strips)
    _attach(model, registry)
    transform = model.get_slice_to_spiral_transform()
    gap = pinned_gap_stage(transform)
    with torch.no_grad():
        diagnostics = gap.pin_diagnostics(model.last_pins)
    assert diagnostics['pin_order_violations'] > 0
    assert diagnostics['pin_residual_max'] > 1.0
    # Pins on rays that carry no swapped pair are still exact: A's quad
    # centres beyond B's theta range and its footprint.
    with torch.no_grad():
        spiral = transform(registry.zyx)
        dr = model.get_dr_per_winding()
        theta, _, shifted = get_theta_and_radii(spiral[:, 1:], dr)
        n = registry.adjusted_n(theta)
        target = dr * (model.effective_pin_targets()[registry.component] + n.float())
    on_a = (registry.kind == pins.PIN_KIND_PATCH) & (torch.linalg.norm(registry.zyx[:, 1:], dim=-1) > 3.0 * DR)
    clean = on_a & (theta > 0.65) & (theta < 0.9) & ~model.merged_pin_mask()
    assert int(clean.sum()) >= 6
    assert (shifted - target)[clean].abs().max() < 2e-3


# ---------------------------------------------------------------------------
# checkpoint round trip and pinned DT targets
# ---------------------------------------------------------------------------


def test_registry_state_dict_round_trip_and_dt_values():
    model = make_model(seed=6, flow_std=1e-3)
    patches, atlas, pcls, strips = build_scene()
    graph, registry = build_registry(model, patches, atlas, pcls, strips)
    restored = pins.PinRegistry.from_state_dict(registry.state_dict(), 'cpu')
    assert restored.num_pins == registry.num_pins
    assert torch.equal(restored.n0, registry.n0)
    assert restored.fingerprint == registry.fingerprint == graph.fingerprint()
    _attach(model, restored)
    transform = model.get_slice_to_spiral_transform()
    with torch.no_grad():
        spiral = transform(restored.zyx)
        dr = model.get_dr_per_winding()
        theta, _, shifted = get_theta_and_radii(spiral[:, 1:], dr)
        target = dr * (model.effective_pin_targets()[restored.component] + restored.adjusted_n(theta).float())
    assert (shifted - target)[~model.merged_pin_mask()].abs().max() < 2e-3

    # Pinned whole-object DT targets: T_g + O_P per patch, snapped by the
    # existing selection policy.
    from dt_targets import compute_patch_dt_target_cache
    values = torch.full([2], float('nan'))
    has = restored.patch_component >= 0
    T = model.effective_pin_targets()
    values[has] = T[restored.patch_component[has]] + restored.patch_offset[has].float()
    assert abs(float(values[atlas.id_to_idx['A']]) - 3.0) < 1e-3
    assert abs(float(values[atlas.id_to_idx['B']]) - 5.0) < 1e-3
    for patch in patches.values():
        patch._dt_target_ijs = np.zeros([0, 2], dtype=np.float32)
        patch._dt_target_block_rc = np.zeros([0, 2], dtype=np.int64)
        patch._dt_target_block_shape = (0, 0)
    cache = compute_patch_dt_target_cache(
        transform, dr, list(patches.values()), atlas, None, 0.25, pinned_values=values)
    assert cache['target_relative'].tolist() == [3, 5]
    assert cache['valid'].all()

    # T sweep: the DT target follows round(T + O_P) (with the floating policy).
    with torch.no_grad():
        model.pin_targets.copy_(model.pin_targets + 0.4)
    values[has] = model.effective_pin_targets()[restored.patch_component[has]] + restored.patch_offset[has].float()
    cache = compute_patch_dt_target_cache(
        transform, dr, list(patches.values()), atlas, None, 0.25, pinned_values=values)
    # The patch component is absolute, so T is fixed and the target unchanged.
    assert cache['target_relative'].tolist() == [3, 5]


def test_one_pinned_transform_serves_several_backwards():
    # The training step evaluates one transform instance per loss family and
    # backwards each family separately (no retain_graph); the pin table must
    # not hold graph nodes shared between those evaluations.
    model = make_model(seed=7, flow_std=1e-3)
    patches, atlas, pcls, strips = build_scene()
    graph, registry = build_registry(model, patches, atlas, pcls, strips)
    _attach(model, registry)
    shared = tuple(o.detach().requires_grad_(True) for o in model.get_shared_transform_tensors())
    assert len(shared) == 4
    transform = model.get_slice_to_spiral_transform(shared=shared)
    points = torch.stack([
        torch.empty([64]).uniform_(45., 125.),
        torch.empty([64]).uniform_(-90., 90.),
        torch.empty([64]).uniform_(-90., 90.),
    ], dim=-1)
    for _ in range(3):
        transform(points).sum().backward()
        transform.inv(points).sum().backward()
    assert shared[3].grad is not None and torch.isfinite(shared[3].grad).all()
    # The pins' graph then propagates once into the model parameters.
    torch.autograd.backward([model.get_shared_transform_tensors()[3]], [shared[3].grad])
    assert model.pin_targets.grad is not None


def test_overlapping_patches_join_one_component():
    # Two patches observing the same sheet (one across the seam), no PCLs:
    # the overlap edges make them one component with the geometric offset,
    # and every pin of both is exact under a deformed model.
    model = make_model(seed=8, flow_std=1e-3, gap_std=0.02)
    patches = {
        'A': make_patch(3, TWO_PI - 0.8, TWO_PI + 0.8, 40.0, 100.0),
        'B': make_patch(3, TWO_PI + 0.2, TWO_PI + 1.4, 60.0, 130.0),   # same sheet, past the seam
        'C': make_patch(5, 1.0, 2.6, 60.0, 130.0),                    # unrelated
    }
    atlas = PatchAtlas(patches, device='cpu')
    pairs = pins.patch_overlap_pairs(atlas, 3.0)
    linked = {(a, b) for a, _, b, _, _ in pairs}
    assert linked == {(atlas.id_to_idx['A'], atlas.id_to_idx['B'])}
    graph = pins.build_pin_graph(
        verified_patches=patches, patch_atlas=atlas, cross_patch_pcls=[],
        unattached_pcl_strips=[], unattached_components=[], unattached_component_edges=[],
        overlap_pairs=pairs)
    assert graph.num_components == 2
    crossing_map = ThetaCrossingMap('cpu')
    atlas.register_theta_topology(crossing_map)
    crossing_map.force_refresh(model.get_slice_to_spiral_transform())
    gap, _, _, _ = model._get_transform_parts(with_pins=False)

    def free_gap(theta, z, slot):
        gaps = gap.get_transformed_winding_radii(theta, z).diff(dim=-1)
        return torch.gather(gaps, -1, slot.clamp(0, gaps.shape[-1] - 1)[:, None]).squeeze(-1)

    registry = pins.finalize_registry(
        graph, intermediate_transform=model.get_slice_to_intermediate_transform(),
        dr=model.get_dr_per_winding(), crossing_map=crossing_map, patch_atlas=atlas,
        footprint_rule=pins.FootprintRule(), free_gap_fn=free_gap,
        min_z=0.0, max_z=192.0, device=torch.device('cpu'))
    assert registry.consistency_report['inconsistent_edges'] == 0
    a, b = atlas.id_to_idx['A'], atlas.id_to_idx['B']
    assert int(registry.patch_component[a]) == int(registry.patch_component[b])
    # Under the identity model both patches' root-frame windings agree with
    # the sheet: T + O_P = 3 for A's root (pre-seam) and 4 for B's (post-seam).
    T = registry.initial_T
    assert abs(float(T[registry.patch_component[a]] + registry.patch_offset[a]) - 3.0) < 0.05
    assert abs(float(T[registry.patch_component[b]] + registry.patch_offset[b]) - 4.0) < 0.05
    _attach(model, registry)
    transform = model.get_slice_to_spiral_transform()
    with torch.no_grad():
        spiral = transform(registry.zyx)
        dr = model.get_dr_per_winding()
        theta, _, shifted = get_theta_and_radii(spiral[:, 1:], dr)
        target = dr * (model.effective_pin_targets()[registry.component] + registry.adjusted_n(theta).float())
        diagnostics = pinned_gap_stage(transform).pin_diagnostics(model.last_pins)
    assert diagnostics['pin_order_violations'] == 0
    assert (shifted - target)[~model.merged_pin_mask()].abs().max() < 2e-3


def test_T_is_estimated_in_canonical_winding_units():
    # A strongly non-identity gap table: intermediate radius / dr is windings
    # off, the canonical shifted winding of the unpinned transform is not.
    model = make_model(seed=9, gap_std=0.01)
    patches, atlas, pcls, strips = build_scene()
    graph, registry = build_registry(model, patches, atlas, pcls, strips)
    with torch.no_grad():
        spiral = model.get_unpinned_slice_to_spiral_transform()(registry.zyx)
        dr = model.get_dr_per_winding()
        theta, _, shifted = get_theta_and_radii(spiral[:, 1:], dr)
        inter = model.get_slice_to_intermediate_transform()(registry.zyx)
        _, _, shifted_intermediate = get_theta_and_radii(inter[:, 1:], dr)
    canonical = (shifted / dr - registry.n0.float())
    intermediate = (shifted_intermediate / dr - registry.n0.float())
    strip = registry.component != int(registry.component[registry.kind == pins.PIN_KIND_PATCH][0])
    T_strip = float(registry.initial_T[registry.component[strip][0]])
    assert abs(T_strip - float(canonical[strip].median())) < 1e-4
    # ...and the gap table really does move the estimate.
    assert abs(float(canonical[strip].median()) - float(intermediate[strip].median())) > 0.05
    _attach(model, registry)
    assert torch.allclose(model.estimate_pin_targets()[~registry.fixed_T],
                          registry.initial_T[~registry.fixed_T], atol=1e-4)


def test_seam_straddling_coincident_pins_merge_consistently():
    # Two pins of one sheet either side of the seam, coincident in (theta, z):
    # raw windings 3 (theta just below 2pi) and 4 (theta just above 0). The
    # merged anchor must target the sheet, not the average of two frames.
    dr = torch.tensor(16.0)
    theta = torch.tensor([TWO_PI - 1e-4, 1e-4])
    z = torch.tensor([50.0, 50.0])
    r = torch.tensor([16.0 * (3 + theta[0] / TWO_PI), 16.0 * (4 + theta[1] / TWO_PI)])
    target_shifted = torch.tensor([16.0 * 3, 16.0 * 4])   # dr (T + n): n differs by one across the seam
    slot = torch.tensor([3, 4])
    eps = torch.tensor([0.1, 0.1])
    eps_z = torch.tensor([5.0, 5.0])
    groups = pins.compute_coincidence_groups(
        theta, z, slot, torch.zeros(2, dtype=torch.int64), torch.tensor([0, 1]), r, eps, eps_z,
        0.0, 100.0, torch.full([2], 16.0), coincidence_frac=0.05, conflict_tolerance=0.1)
    assert groups.num_groups == 1 and groups.num_conflicts == 0
    table = pins.PinTable(z, theta, r, target_shifted, slot, eps, eps_z, 12, 0.0, 100.0, dr, groups=groups)
    for theta_q in (TWO_PI - 1e-4, 1e-4):
        R, S, w, valid = table.anchors(torch.tensor([theta_q]), torch.tensor([50.0]))
        k = int(torch.nonzero(valid[0])[0])
        assert k == (3 if theta_q > 1.0 else 4)
        assert abs(float(S[0, k]) - 16.0 * (k + theta_q / TWO_PI)) < 1e-3
        assert abs(float(R[0, k]) - float(r.mean())) < 1e-6


def test_kernel_gradient_reaches_pin_coordinates():
    dr = torch.tensor(16.0)
    theta = torch.tensor([1.0, 1.02], requires_grad=True)
    z = torch.tensor([50.0, 51.0], requires_grad=True)
    r = torch.tensor([70.0, 70.5])
    target = torch.tensor([64.0, 64.0])
    table = pins.PinTable(z, theta, r, target, torch.tensor([4, 4]), torch.full([2], 0.1),
                          torch.full([2], 5.0), 12, 0.0, 100.0, dr)
    R, S, w, valid = table.anchors(torch.tensor([1.01]), torch.tensor([50.5]))
    R[0, 4].backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all() and theta.grad.abs().sum() > 0
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_subset_change_rebuilds_coincidence_groups():
    model = make_model(seed=10, flow_std=1e-3)
    model.cfg['sample_count_pins'] = 40
    patches, atlas, pcls, strips = build_scene()
    graph, registry = build_registry(model, patches, atlas, pcls, strips)
    _attach(model, registry)
    torch.manual_seed(0)
    model.get_slice_to_spiral_transform(shared=tuple(
        o.detach().requires_grad_(True) for o in model.get_shared_transform_tensors()))
    first = model._pin_view['indices'].clone()
    rebuilds = model._pin_rebuilds
    model.get_slice_to_spiral_transform(shared=tuple(
        o.detach().requires_grad_(True) for o in model.get_shared_transform_tensors()))
    assert not torch.equal(first, model._pin_view['indices'])
    assert model._pin_rebuilds == rebuilds + 1


def test_coincidence_groups_follow_moving_pins():
    # Full registry (no subsampling): two coincident pins are merged; after
    # the pins move apart the next table must split them again (groups are
    # rebuilt every model_pin_rebin_interval steps).
    model = make_model(seed=11)
    patches, atlas, pcls, strips = build_scene()
    graph, registry = build_registry(model, patches, atlas, pcls, strips)
    _attach(model, registry)
    model.get_slice_to_spiral_transform()
    assert int(model.merged_pin_mask().sum()) == 2
    merged = torch.nonzero(model.merged_pin_mask()).flatten()
    rebuilds = model._pin_rebuilds
    # Move one member of the pair 30 voxels away in z (the registry is data;
    # this emulates the flow moving it).
    with torch.no_grad():
        model.pin_registry.zyx[merged[0], 0] += 30.0
    model.get_slice_to_spiral_transform()
    assert model._pin_rebuilds == rebuilds + 1
    assert int(model.merged_pin_mask().sum()) == 0
    # With a long rebin interval the stale groups would have persisted.
    model.cfg['model_pin_rebin_interval'] = 1000
    model.get_slice_to_spiral_transform()
    with torch.no_grad():
        model.pin_registry.zyx[merged[0], 0] -= 30.0
    model.get_slice_to_spiral_transform()
    assert int(model.merged_pin_mask().sum()) == 2   # bin crossing forces safe rebuild


def test_checkpoint_resume_preserves_pin_frame_after_seam_crossing():
    from types import SimpleNamespace
    from fit_spiral import FitContext

    model = make_model()
    graph = pins.PinGraph()
    a = graph.add_point(spiral_point(4, 6.2, 50), 'a')
    b = graph.add_point(spiral_point(5, 0.1, 50), 'b')
    graph.add_chain_edge(a, b)

    def finalize(model):
        return pins.finalize_registry(
            graph, intermediate_transform=model.get_slice_to_intermediate_transform(),
            canonical_transform=model.get_unpinned_slice_to_spiral_transform(),
            dr=model.get_dr_per_winding(), crossing_map=None, patch_atlas=None,
            footprint_rule=pins.FootprintRule(),
            free_gap_fn=lambda theta, z, slot: torch.full_like(theta, DR),
            min_z=0., max_z=192., device='cpu')

    registry = finalize(model)
    _attach(model, registry)
    optimizer = torch.optim.AdamW([model.pin_targets])
    model.pin_targets.sum().backward()
    optimizer.step()
    with torch.no_grad():
        # The inverse linear stage rotates both pins by +0.2 radians, taking
        # the component root across theta=0 after the registry was created.
        model.linear_logits[:, 0, 1] = -0.2 / model.linear_logits_scale
        model.linear_logits[:, 1, 0] = 0.2 / model.linear_logits_scale
        rebuilt = finalize(model)
        assert registry.adjusted_n(rebuilt.theta0).tolist() == [1, 1]
        assert rebuilt.n0.tolist() == [0, 0]
        expected = model.get_slice_to_spiral_transform()(registry.zyx)
    saved = {
        'spiral_and_transform': model.state_dict(),
        'pin_registry': registry.state_dict(),
        'optimiser': optimizer.state_dict(),
    }

    resumed = make_model()
    resumed.init_pin_targets(graph.num_components)
    context = FitContext.__new__(FitContext)
    context.spiral_and_transform = resumed
    context.pin_graph = graph
    context.pin_target_params = [resumed.pin_targets]
    context.optimiser = torch.optim.AdamW([resumed.pin_targets])
    context.dist = SimpleNamespace(is_main_process=False)
    context.config = {'model_pins_warmup_steps': 0}
    context.start_iteration = 600
    context.device = torch.device('cpu')
    context.flow_min_corner_spiral_zyx = resumed.flow_min_corner_zyx
    context.flow_max_corner_spiral_zyx = resumed.flow_max_corner_zyx
    context.theta_crossing_map = None
    context.patch_atlas = None
    state, optimizer_state, _ = context._adapt_checkpoint_for_pins(
        saved, saved['spiral_and_transform'], saved['optimiser'])
    resumed.load_state_dict(state)
    context.optimiser.load_state_dict(optimizer_state)
    torch.testing.assert_close(
        context.optimiser.state[resumed.pin_targets]['exp_avg'],
        optimizer.state[model.pin_targets]['exp_avg'])
    context._finalize_pin_registry()

    assert torch.equal(resumed.pin_registry.theta0, registry.theta0)
    assert torch.equal(resumed.pin_registry.n0, registry.n0)
    assert torch.equal(resumed.pin_targets, model.pin_targets)
    with torch.no_grad():
        actual = resumed.get_slice_to_spiral_transform()(registry.zyx)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pin_graph_and_registry_exclude_points_outside_flow_z_domain():
    """PCL/strip points and patch quad centres outside the flow z domain are
    not pins, and chains are cut there (whole-scroll PCLs extend far beyond a
    z-range fit; the transform is undefined outside its box)."""
    model = make_model()
    patches, atlas, pcls, strips = build_scene()
    # Push one cross-patch PCL point far outside z in [0, 192].
    far = None
    for pcl in pcls:
        for point in pcl['points'].values():
            if 'on_patch' not in point:
                point['zyx'] = np.array([900.0, point['zyx'][1], point['zyx'][2]], dtype=np.float32)
                far = point
                break
        if far is not None:
            break
    assert far is not None
    graph_all = pins.build_pin_graph(
        verified_patches=patches, patch_atlas=atlas, cross_patch_pcls=pcls,
        unattached_pcl_strips=strips, unattached_components=[[0]],
        unattached_component_edges=[[]], patch_grid_stride=1)
    graph = pins.build_pin_graph(
        verified_patches=patches, patch_atlas=atlas, cross_patch_pcls=pcls,
        unattached_pcl_strips=strips, unattached_components=[[0]],
        unattached_component_edges=[[]], patch_grid_stride=1, z_range=(0.0, 192.0))
    point_nodes = [n for n in graph.nodes if n.kind != 'patch']
    assert len(point_nodes) == len([n for n in graph_all.nodes if n.kind != 'patch']) - 1
    assert all(0.0 <= float(n.zyx[0]) <= 192.0 for n in point_nodes)
    assert len(graph.edges) < len(graph_all.edges)
    # Registry built from the filtered graph carries no out-of-domain pin.
    _, registry = build_registry(model, patches, atlas, pcls, strips, z_range=(0.0, 192.0))
    assert registry.num_pins > 0
    assert bool((registry.zyx[:, 0] >= 0.0).all() and (registry.zyx[:, 0] <= 192.0).all())
