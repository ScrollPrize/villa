"""The pair-agreement warm-up loss (pinned_spiral_plan.md, "preparing the warm-up")."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pins
from losses import get_pair_agreement_loss, pair_agreement_residuals
from sample_spiral import get_theta_and_radii
from test_pins_transform import make_model, build_scene, build_registry

TWO_PI = 2 * math.pi


# ---------------------------------------------------------------- residuals

def test_residual_is_distance_to_nearest_integer():
    w_a = torch.tensor([3.0, 3.0, 3.0, 3.3, 3.3, 2.7, 10.49, 10.51])
    w_b = torch.tensor([3.0, 4.0, 1.0, 3.0, 4.0, 3.4, 10.0, 10.0])
    r = pair_agreement_residuals(w_a, w_b)
    expected = torch.tensor([0.0, 0.0, 0.0, 0.3, 0.3, -0.7 + 1.0, 0.49, -0.49])
    assert torch.allclose(r, expected, atol=1e-6)
    assert bool((r > -0.5).all()) and bool((r <= 0.5).all())
    # Antisymmetric in the pair order.
    assert torch.allclose(pair_agreement_residuals(w_b, w_a), -r, atol=1e-6)


def test_residual_gradient_pushes_toward_nearest_integer():
    for value, sign in ((3.3, -1.0), (2.7, +1.0), (5.1, -1.0), (4.9, +1.0)):
        w_a = torch.tensor([value], requires_grad=True)
        w_b = torch.tensor([3.0])
        loss = torch.nn.functional.relu(pair_agreement_residuals(w_a, w_b).abs() - 0.05).mean()
        loss.backward()
        # Descent moves w_a toward the nearest integer difference.
        assert math.copysign(1.0, -float(w_a.grad)) == sign
    # Inside the margin the hinge is flat.
    w_a = torch.tensor([3.02], requires_grad=True)
    loss = torch.nn.functional.relu(pair_agreement_residuals(w_a, torch.tensor([3.0])).abs() - 0.05).mean()
    loss.backward()
    assert float(w_a.grad) == 0.0


def test_residual_gradcheck_double():
    w_a = torch.tensor([3.31, 2.68, 7.2], dtype=torch.float64, requires_grad=True)
    w_b = torch.tensor([3.0, 3.0, 4.9], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: torch.nn.functional.relu(pair_agreement_residuals(a, b).abs() - 0.05).sum(),
        (w_a, w_b))


# ---------------------------------------------------------------- pair builder

def _grid(origin, n, spacing, z0=0.0):
    ys, xs = torch.meshgrid(torch.arange(n) * spacing, torch.arange(n) * spacing, indexing='ij')
    return torch.stack([torch.full_like(ys, z0), ys + origin[0], xs + origin[1]], -1).reshape(-1, 3).float()


def test_cross_component_pairs_are_cross_component_within_tolerance():
    a = _grid((0.0, 0.0), 6, 4.0)            # component 0 (patch 0)
    b = _grid((0.0, 10.0), 6, 4.0)           # component 1 (patch 1): interleaves with a in x
    c = _grid((500.0, 500.0), 6, 4.0)        # component 2 (patch 2): far away
    zyx = torch.cat([a, b, c])
    patch_index = torch.cat([torch.zeros(36), torch.ones(36), torch.full([36], 2)]).long()
    component = patch_index.clone()
    pairs = pins.cross_component_pairs(zyx, patch_index, component, tolerance=8.0, stride=1)
    assert pairs.dtype == torch.int64 and pairs.shape[1] == 2 and pairs.shape[0] > 0
    i, j = pairs[:, 0], pairs[:, 1]
    assert bool((component[i] != component[j]).all())
    assert bool(((zyx[i] - zyx[j]).norm(dim=-1) <= 8.0 + 1e-6).all())
    assert not bool(((patch_index[i] == 2) | (patch_index[j] == 2)).any())
    # Sorted lexicographically and unique.
    key = i * zyx.shape[0] + j
    assert bool((key[1:] > key[:-1]).all())
    # Brute force agrees.
    d = torch.cdist(zyx, zyx)
    cross = component[:, None] != component[None, :]
    expected = int((torch.triu(cross & (d <= 8.0), diagonal=1)).sum())
    assert pairs.shape[0] == expected


def test_cross_component_pairs_stride_cap_and_degenerate_inputs():
    a = _grid((0.0, 0.0), 8, 3.0)
    b = _grid((0.0, 1.5), 8, 3.0)
    zyx = torch.cat([a, b])
    patch_index = torch.cat([torch.zeros(64), torch.ones(64)]).long()
    component = patch_index.clone()
    full = pins.cross_component_pairs(zyx, patch_index, component, tolerance=5.0, stride=1)
    strided = pins.cross_component_pairs(zyx, patch_index, component, tolerance=5.0, stride=2)
    assert 0 < strided.shape[0] < full.shape[0]
    assert bool((strided % 2 == 0).all())
    capped = pins.cross_component_pairs(zyx, patch_index, component, tolerance=5.0, stride=1, max_pairs=10, seed=3)
    assert capped.shape[0] == 10
    again = pins.cross_component_pairs(zyx, patch_index, component, tolerance=5.0, stride=1, max_pairs=10, seed=3)
    assert torch.equal(capped, again)
    other = pins.cross_component_pairs(zyx, patch_index, component, tolerance=5.0, stride=1, max_pairs=10, seed=4)
    assert not torch.equal(capped, other)
    # Every capped pair is a real pair.
    full_keys = set((full[:, 0] * zyx.shape[0] + full[:, 1]).tolist())
    assert set((capped[:, 0] * zyx.shape[0] + capped[:, 1]).tolist()) <= full_keys
    # One component: nothing to pair. Non-patch pins (patch_index -1) are skipped.
    assert pins.cross_component_pairs(zyx, patch_index, torch.zeros_like(component), tolerance=5.0, stride=1).shape[0] == 0
    assert pins.cross_component_pairs(zyx, torch.full_like(patch_index, -1), component, tolerance=5.0, stride=1).shape[0] == 0
    assert pins.cross_component_pairs(zyx[:0], patch_index[:0], component[:0]).shape == (0, 2)


# ---------------------------------------------------------------- the loss

def _ideal_points(winding, theta, dr=20.0, z=50.0):
    """Scroll points on the ideal Archimedean spiral (identity transform):
    radius = dr * (winding + theta / 2pi), umbilicus at the origin."""
    radius = dr * (winding + theta / TWO_PI)
    return torch.stack([torch.full_like(theta, z), radius * torch.sin(theta), radius * torch.cos(theta)], -1)


def test_loss_zero_on_ideal_spiral_including_seam_and_adjacent_sheets():
    dr = torch.tensor(20.0)
    identity = lambda zyx: zyx
    theta = torch.linspace(0.05, TWO_PI - 0.05, 40)
    same_sheet = (_ideal_points(3.0, theta), _ideal_points(3.0, theta + 0.02))
    adjacent = (_ideal_points(3.0, theta), _ideal_points(4.0, theta))
    # Straddling the theta = 0 seam: the shifted radius jumps by one winding.
    seam = (_ideal_points(3.0, torch.tensor([TWO_PI - 0.01])), _ideal_points(3.0, torch.tensor([0.01])))
    for a, b in (same_sheet, adjacent, seam):
        loss, residual = get_pair_agreement_loss(a, b, identity, dr, margin=0.0)
        assert float(residual.abs().max()) < 1e-3, residual
        assert float(loss) < 1e-3
    # Half a sheet apart: residual 0.5 in magnitude.
    loss, residual = get_pair_agreement_loss(_ideal_points(3.0, theta), _ideal_points(3.5, theta), identity, dr, margin=0.05)
    assert torch.allclose(residual.abs(), torch.full_like(residual, 0.5), atol=1e-3)
    assert abs(float(loss) - 0.45) < 1e-3
    # No pairs: zero loss, empty residual, no NaN.
    loss, residual = get_pair_agreement_loss(theta.new_zeros([0, 3]), theta.new_zeros([0, 3]), identity, dr, margin=0.0)
    assert float(loss) == 0.0 and residual.numel() == 0


def test_loss_matches_free_map_windings_and_chunks():
    dr = torch.tensor(20.0)
    identity = lambda zyx: zyx
    theta = torch.rand(300) * TWO_PI
    a = _ideal_points(3.0, theta)
    b = _ideal_points(3.0 + 0.37, torch.rand(300) * TWO_PI)
    loss, residual = get_pair_agreement_loss(a, b, identity, dr, margin=0.0, chunk_size=64)
    _, _, s_a = get_theta_and_radii(a[..., 1:], dr)
    _, _, s_b = get_theta_and_radii(b[..., 1:], dr)
    expected = pair_agreement_residuals(s_a / dr, s_b / dr)
    assert torch.allclose(residual, expected, atol=1e-5)
    assert torch.allclose(residual.abs(), torch.full_like(residual, 0.37), atol=1e-3)
    loss_whole, _ = get_pair_agreement_loss(a, b, identity, dr, margin=0.0)
    assert abs(float(loss) - float(loss_whole)) < 1e-6


def test_loss_on_model_registry_pairs_descends_on_gap_logits():
    # The scene's scroll is the ideal spiral, so under the identity model
    # every cross-component pair differs by a whole winding: zero loss. A
    # gap-logit perturbation breaks that, and a step along the gradient on
    # the shared leaves reduces the loss.
    model = make_model(seed=1)
    patches, atlas, pcls, strips = build_scene()
    # Without the linking PCLs the two patches are separate components
    # (with them they share one and there is nothing to pair).
    _, registry = build_registry(model, patches, atlas, [], strips)
    pairs = pins.cross_component_pairs(registry.zyx, registry.patch_index, registry.component,
                                       tolerance=60.0, stride=1)
    assert pairs.shape[0] > 0
    assert bool((registry.component[pairs[:, 0]] != registry.component[pairs[:, 1]]).all())
    zyx_a, zyx_b = registry.zyx[pairs[:, 0]], registry.zyx[pairs[:, 1]]

    shared = tuple(x.detach().requires_grad_(True) for x in model.get_shared_transform_tensors())
    transform = model.get_unpinned_slice_to_spiral_transform(shared=shared)
    loss, residual = get_pair_agreement_loss(zyx_a, zyx_b, transform, shared[0], margin=0.0)
    assert float(residual.abs().max()) < 2e-2, float(residual.abs().max())

    logits = shared[2]
    with torch.no_grad():
        logits[..., 3, :] += 3.0     # stretch one winding gap on every ray
    margin = 0.05
    loss, residual = get_pair_agreement_loss(zyx_a, zyx_b, transform, shared[0], margin=margin)
    assert float(loss) > 1e-3
    grad = torch.autograd.grad(loss, logits)[0]
    assert torch.isfinite(grad).all() and float(grad.abs().sum()) > 0
    decreased = False
    for eps in (1e-4, 1e-5, 1e-6, 1e-7):
        with torch.no_grad():
            logits -= eps * grad / grad.abs().max()
        loss_after, _ = get_pair_agreement_loss(zyx_a, zyx_b, transform, shared[0], margin=margin)
        with torch.no_grad():
            logits += eps * grad / grad.abs().max()
        if float(loss_after) < float(loss):
            decreased = True
            break
    assert decreased


def test_loss_with_pins_active_uses_free_map():
    # With the pins switched on, the unpinned transform (what the loss
    # evaluates) is the same object as before activation for the same leaves.
    model = make_model(seed=2, gap_std=0.2)
    patches, atlas, pcls, strips = build_scene()
    _, registry = build_registry(model, patches, atlas, [], strips)
    pairs = pins.cross_component_pairs(registry.zyx, registry.patch_index, registry.component,
                                       tolerance=60.0, stride=1)
    zyx_a, zyx_b = registry.zyx[pairs[:, 0]], registry.zyx[pairs[:, 1]]
    dr = model.get_dr_per_winding().detach()
    free_before = model.get_unpinned_slice_to_spiral_transform()
    _, before = get_pair_agreement_loss(zyx_a, zyx_b, free_before, dr, margin=0.0)
    model.set_pin_registry(registry)
    model.pins_active = True
    shared = tuple(x.detach().requires_grad_(True) for x in model.get_shared_transform_tensors())
    assert len(shared) > 3
    free_after = model.get_unpinned_slice_to_spiral_transform(shared=shared)
    _, after = get_pair_agreement_loss(zyx_a, zyx_b, free_after, shared[0], margin=0.0)
    assert torch.allclose(before, after, atol=1e-5)
    assert model.pins_active  # restored
    pinned = model.get_slice_to_spiral_transform(shared=shared)
    _, pinned_residual = get_pair_agreement_loss(zyx_a, zyx_b, pinned, shared[0], margin=0.0)
    # The pinned map differs from the free map here (random gap logits).
    assert not torch.allclose(before, pinned_residual, atol=1e-3)
