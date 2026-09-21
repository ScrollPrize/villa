"""Stage 3: the pin strain loss (pinned_spiral_plan.md)."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pins
from losses import get_pin_strain_loss
from test_pins_transform import make_model, build_scene, build_registry, _attach

TWO_PI = 2 * math.pi


def pinned_scene():
    model = make_model(flow_std=0.02, gap_std=0.3)
    patches, atlas, pcls, strips = build_scene()
    _, registry = build_registry(model, patches, atlas, pcls, strips)
    _attach(model, registry)
    return model


def test_strain_zero_iff_free_map_satisfies_and_gradient_sign():
    model = pinned_scene()
    shared = tuple(x.detach().requires_grad_(True) for x in model.get_shared_transform_tensors())
    transform = model.get_slice_to_spiral_transform(shared=shared)
    gap = transform.inv.parts[0]
    pins_leaf = shared[3]
    # Move every pin's radius onto the free map's inverse of its target: the
    # pins are then not needed and the strain is identically zero.
    z, theta, r, target_shifted = pins_leaf.detach().unbind(-1)
    theta_norm = theta / TWO_PI
    table = gap.get_transformed_winding_radii(theta, z).detach()
    r_on_free = pins.unpinned_map_inverse(target_shifted + gap.dr_per_winding.detach() * theta_norm,
                                          table, gap.dr_per_winding.detach(), theta_norm)
    satisfied = torch.stack([z, theta, r_on_free, target_shifted], -1).requires_grad_(True)
    loss, strain = get_pin_strain_loss(satisfied, gap, shared[0], margin=0.0)
    assert float(strain.abs().max()) < 1e-4
    assert float(loss) < 1e-4
    # Perturb one gap logit: the free map moves, the strain becomes positive
    # and a gradient step on the logits reduces it.
    logits = shared[2]
    with torch.no_grad():
        logits[..., 3, 10] += 2.0
    # The hinge margin matters here: at margin 0 every satisfied pin sits on
    # the |strain| kink, and since gaps accumulate outward any logit change
    # gives them nonzero strain, so the loss would rise in both directions.
    margin = 0.025
    loss, strain = get_pin_strain_loss(satisfied, gap, shared[0], margin=margin)
    assert float(strain.abs().max()) > 1e-3
    grad = torch.autograd.grad(loss, logits)[0]
    assert float(grad.abs().sum()) > 0
    # Descent direction check by a short line search: the calibrated softplus
    # scale makes the free table very sensitive to the scaled logits, so the
    # step has to be tiny in logit units.
    decreased = False
    for eps in (1e-6, 1e-7, 1e-8):
        with torch.no_grad():
            logits -= eps * grad / grad.abs().max()
        loss_after, _ = get_pin_strain_loss(satisfied, gap, shared[0], margin=margin)
        with torch.no_grad():
            logits += eps * grad / grad.abs().max()
        if float(loss_after) < float(loss):
            decreased = True
            break
    assert decreased


def test_strain_is_deviation_from_component_winding_and_detaches_targets():
    model = pinned_scene()
    shared = tuple(x.detach().requires_grad_(True) for x in model.get_shared_transform_tensors())
    transform = model.get_slice_to_spiral_transform(shared=shared)
    gap = transform.inv.parts[0]
    pins_leaf = shared[3]
    loss, strain = get_pin_strain_loss(pins_leaf, gap, shared[0], margin=0.0)
    # Reference: the free map's shifted winding at each pin minus T + n.
    free = model.get_unpinned_slice_to_spiral_transform()
    with torch.no_grad():
        spiral = free(model.pin_registry.zyx[model._pin_view['indices']]
                      if model._pin_view['indices'] is not None else model.pin_registry.zyx)
        th = torch.atan2(spiral[:, 1], spiral[:, 2]) % TWO_PI
        w_free = spiral[:, 1:].norm(dim=-1) / shared[0] - th / TWO_PI
        expected = w_free - pins_leaf[:, 3] / shared[0]
    torch.testing.assert_close(strain, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(loss, expected.abs().mean(), atol=1e-4, rtol=1e-4)
    # Detached targets: no gradient into the target column of the pins leaf;
    # the position columns and the gap logits do receive gradient.
    g_pins, g_logits = torch.autograd.grad(loss, (pins_leaf, shared[2]))
    assert float(g_pins[:, 3].abs().max()) == 0.0
    assert float(g_pins[:, :3].abs().sum()) > 0
    assert float(g_logits.abs().sum()) > 0
    loss_t, _ = get_pin_strain_loss(pins_leaf, gap, shared[0], margin=0.0, detach_targets=False)
    g_pins_t = torch.autograd.grad(loss_t, pins_leaf)[0]
    assert float(g_pins_t[:, 3].abs().sum()) > 0
    # Hinge: a margin above every strain gives zero loss.
    loss_m, _ = get_pin_strain_loss(pins_leaf, gap, shared[0], margin=float(strain.abs().max()) + 1e-3)
    assert float(loss_m) == 0.0


class _FreeTableStage:
    """A stand-in gap stage: a differentiable [rays, windings] free table."""

    def __init__(self, gaps, dr):
        self.gaps = gaps
        self.dr_per_winding = dr

    def get_transformed_winding_radii(self, theta, z):
        zero = self.dr_per_winding * theta[:, None] / TWO_PI
        return torch.cat([zero, zero + torch.cumsum(self.gaps, dim=-1)], dim=-1)


def test_strain_gradcheck_double():
    torch.manual_seed(0)
    rays, windings = 5, 8
    dr = torch.tensor(16.0, dtype=torch.float64, requires_grad=True)
    gaps = (16.0 * (1.0 + 0.4 * (torch.rand(rays, windings - 1, dtype=torch.float64) - 0.5))).requires_grad_(True)
    theta = torch.rand(rays, dtype=torch.float64) * TWO_PI
    z = torch.rand(rays, dtype=torch.float64) * 100
    r = 16.0 * (torch.rand(rays, dtype=torch.float64) * 5 + 1)
    target = 16.0 * torch.randint(1, 6, [rays]).to(torch.float64) + torch.randn(rays, dtype=torch.float64)
    pins_leaf = torch.stack([z, theta, r, target], -1).requires_grad_(True)

    def fn(pins_leaf, gaps, dr):
        return get_pin_strain_loss(pins_leaf, _FreeTableStage(gaps, dr), dr, margin=0.01,
                                   detach_targets=False)[0]

    assert torch.autograd.gradcheck(fn, (pins_leaf, gaps, dr), atol=1e-6, rtol=1e-5)
