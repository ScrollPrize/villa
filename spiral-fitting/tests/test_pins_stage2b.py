"""Performance paths retain the radius-space Stage 2a reference semantics."""
import dataclasses
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pins
from test_pins_transform import make_model, build_scene, build_registry, _attach
from transforms import ray_specialized_spiral_to_scroll


@pytest.fixture
def no_gpu_fuser():
    """The flow backward is torch.jit-scripted; its NVRTC fusion needs a
    CUDA runtime matching torch and is irrelevant to these comparisons."""
    previous = torch._C._jit_can_fuse_on_gpu()
    torch._C._jit_override_can_fuse_on_gpu(False)
    yield
    torch._C._jit_override_can_fuse_on_gpu(previous)


def scene(device='cpu'):
    model = make_model()
    patches, atlas, pcls, strips = build_scene()
    _, registry = build_registry(model, patches, atlas, pcls, strips)
    if device != 'cpu':
        # The corners and umbilicus tables are plain tensors, not buffers.
        model = model.to(device)
        model.flow_min_corner_zyx = model.flow_min_corner_zyx.to(device)
        model.flow_max_corner_zyx = model.flow_max_corner_zyx.to(device)
        model.umbilicus_transform._yx = model.umbilicus_transform._yx.to(device)
        model.umbilicus_transform._z = model.umbilicus_transform._z.to(device)
    _attach(model, registry)
    return model


def test_exact_sampling_budget_and_component_coverage():
    model = scene()
    for budget in (1, 17, 40, 101):
        torch.manual_seed(42)
        subset = model.sample_pin_subset(budget)
        indices = subset[0]
        expected = min(model.pin_registry.num_pins, max(budget, model.pin_registry.num_components))
        assert indices.numel() == expected
        assert indices.unique().numel() == expected
        assert torch.equal(model.pin_registry.component[indices].unique(), model.pin_registry.component.unique())
        torch.manual_seed(42)
        assert torch.equal(indices, model.sample_pin_subset(budget)[0])
    model.cfg['sample_count_pins'] = 17
    assert model.compute_pins(full=False).shape[0] == 17
    assert model.compute_pins(full=True).shape[0] == model.pin_registry.num_pins


def test_amortized_layout_and_fresh_leaf_graphs():
    model = scene()
    model.cfg['model_pin_rebin_interval'] = 3
    shared = tuple(x.detach().requires_grad_(True) for x in model.get_shared_transform_tensors())
    transform = model.get_slice_to_spiral_transform(shared=shared)
    first = model._pin_layout_cache['classes']
    rebuilds = model._pin_rebuilds
    model.get_slice_to_spiral_transform(shared=shared)
    assert model._pin_layout_cache['classes'] is first
    assert model._pin_rebuilds == rebuilds
    query = model.pin_registry.zyx[::10].clone()
    for _ in range(2):
        transform(query).square().mean().backward()
    assert torch.isfinite(shared[3].grad).all()
    model.get_slice_to_spiral_transform(shared=shared)
    model.get_slice_to_spiral_transform(shared=shared)
    assert model._pin_rebuilds == rebuilds + 1
    with torch.no_grad():
        model.pin_targets.add_(1)
    model.get_slice_to_spiral_transform()
    assert model._pin_rebuilds == rebuilds + 2


def test_ray_reuse_values_and_gradients():
    model = scene()
    shared = tuple(x.detach().requires_grad_(True) for x in model.get_shared_transform_tensors())
    transform = model.get_slice_to_spiral_transform(shared=shared)
    theta = torch.tensor([0.42, 1.12, 3.4], requires_grad=True)
    z = torch.tensor([52., 66., 82.], requires_grad=True)
    ids = torch.arange(3).repeat_interleave(5)
    radius = torch.linspace(30, 145, 15, requires_grad=True)
    fast = ray_specialized_spiral_to_scroll(transform, radius, theta, z, ids, theta.sin(), theta.cos())
    slow = transform.inv(torch.stack([z[ids], theta.sin()[ids] * radius, theta.cos()[ids] * radius], dim=-1))
    assert fast is not None
    torch.testing.assert_close(fast, slow, atol=3e-5, rtol=3e-5)
    inputs = (radius, theta, z, shared[3])
    fg = torch.autograd.grad(fast.square().mean(), inputs)
    sg = torch.autograd.grad(slow.square().mean(), inputs)
    for actual, expected in zip(fg, sg):
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=3e-4)


def test_refresh_footprints_reproduces_grid_and_tracks_deformation():
    model = scene()
    value = model.compute_pins(full=True)
    model.refresh_pin_footprints(value)
    initial = model._pin_view['eps_z'].clone()
    modified = value.detach().clone()
    modified[:, 0] *= 1.1
    model.refresh_pin_footprints(modified)
    updated = model._pin_view['eps_z']
    assert (updated >= initial).all()
    assert (updated > initial).any()
    assert model._pin_footprint_drift <= 0.10001


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_piecewise_values_and_backward(monkeypatch):
    x = torch.tensor([[0., 2., 5., 0.], [0., 1., 4., 7.]], device='cuda', requires_grad=True)
    y = torch.tensor([[0., 3., 6., 0.], [0., 2., 3., 8.]], device='cuda', requires_grad=True)
    q = torch.tensor([[-1., 0., 1.5, 3., 7.], [-1., 0., 2., 6., 9.]], device='cuda', requires_grad=True)
    counts = torch.tensor([3, 4], device='cuda')
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '1')
    actual = pins.piecewise_linear_map(q, x, y, counts)
    ga = torch.autograd.grad(actual.square().sum(), (q, x, y))
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '0')
    expected = pins.piecewise_linear_map(q, x, y, counts)
    ge = torch.autograd.grad(expected.square().sum(), (q, x, y))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_pinned_transform_matches_eager(monkeypatch, no_gpu_fuser):
    """Whole pinned transform, generic and ray-specialised, Triton vs eager.

    Values agree to fp32 tolerance; gradients into the shared leaves are
    compared at fp-association tolerance relative to each leaf's magnitude
    (the fused backward accumulates with atomics).
    """
    torch.manual_seed(0)
    theta = torch.tensor([0.42, 1.12, 3.4, 6.2], device='cuda')
    z = torch.tensor([52., 66., 82., 120.], device='cuda')
    ids = torch.arange(4, device='cuda').repeat_interleave(6)
    radius = torch.linspace(5, 160, 24, device='cuda')
    points = torch.stack([z[ids], theta.sin()[ids] * radius, theta.cos()[ids] * radius], dim=-1)
    results = {}
    for flag in ('1', '0'):
        monkeypatch.setenv('FIT_SPIRAL_TRITON', flag)
        model = scene('cuda')
        shared = tuple(x.detach().requires_grad_(True) for x in model.get_shared_transform_tensors())
        transform = model.get_slice_to_spiral_transform(shared=shared)
        p = points.clone().requires_grad_(True)
        th, zz, rr = (x.clone().requires_grad_(True) for x in (theta, z, radius))
        outputs = {
            'forward': (transform(p), (p,)),
            'inverse': (transform.inv(transform(points).detach()), ()),
            'rays': (ray_specialized_spiral_to_scroll(transform, rr, th, zz, ids, th.sin(), th.cos()), (rr, th, zz)),
        }
        assert outputs['rays'][0] is not None
        for name, (value, inputs) in outputs.items():
            grads = torch.autograd.grad(value.square().mean(), inputs + shared)
            results[flag, name] = (value.detach(), [g.detach() for g in grads])
    torch.testing.assert_close(results['1', 'inverse'][0], points, atol=1e-3, rtol=1e-5)
    for name in ('forward', 'inverse', 'rays'):
        (fused, fused_grads), (eager, eager_grads) = results['1', name], results['0', name]
        torch.testing.assert_close(fused, eager, atol=1e-4, rtol=1e-5)
        for index, (a, e) in enumerate(zip(fused_grads, eager_grads)):
            scale = float(e.abs().max().clamp(min=1.0))
            # Gradients into ray/pin positions (the sample inputs and the
            # pins leaf) pass through the singular anchor kernel's
            # derivative, which fp32 resolves only to ~1e-2 relative in
            # either implementation (both sit far from a float64 reference);
            # dr, the linear and gap logits are compared at fp-association
            # tolerance.
            positional = index < len(fused_grads) - 4 or index == len(fused_grads) - 1
            torch.testing.assert_close(a, e, atol=1e-5 * scale, rtol=1e-2 if positional else 1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_tridiagonal_solve_matches_eager():
    import gap_triton
    torch.manual_seed(0)
    n, w = 37, 19
    sub = (torch.rand(n, w, device='cuda') * 0.4).requires_grad_(True)
    sup = (torch.rand(n, w, device='cuda') * 0.4).requires_grad_(True)
    diag = (1.0 + torch.rand(n, w, device='cuda')).requires_grad_(True)
    rhs = torch.randn(n, w, device='cuda', requires_grad=True)
    fused = gap_triton.tridiagonal_solve(sub, diag, sup, rhs)
    eager = gap_triton._thomas_eager(sub, diag, sup, rhs)
    torch.testing.assert_close(fused, eager, atol=1e-5, rtol=1e-5)
    weight = torch.randn_like(fused)
    gf = torch.autograd.grad((fused * weight).sum(), (sub, diag, sup, rhs))
    ge = torch.autograd.grad((eager * weight).sum(), (sub, diag, sup, rhs))
    for a, e in zip(gf, ge):
        torch.testing.assert_close(a, e, atol=1e-4, rtol=1e-4)
    # The solution satisfies the system.
    x = fused.detach()
    lhs = diag.detach() * x
    lhs[:, 1:] += sub.detach()[:, 1:] * x[:, :-1]
    lhs[:, :-1] += sup.detach()[:, :-1] * x[:, 1:]
    torch.testing.assert_close(lhs, rhs.detach(), atol=1e-5, rtol=1e-5)


def test_tridiagonal_solve_gradcheck_double():
    import gap_triton
    torch.manual_seed(1)
    n, w = 5, 7
    args = [(torch.rand(n, w, dtype=torch.float64) * 0.4).requires_grad_(True),
            (1.0 + torch.rand(n, w, dtype=torch.float64)).requires_grad_(True),
            (torch.rand(n, w, dtype=torch.float64) * 0.4).requires_grad_(True),
            torch.randn(n, w, dtype=torch.float64, requires_grad=True)]
    assert torch.autograd.gradcheck(gap_triton.tridiagonal_solve, args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_anchor_sums_match_eager(monkeypatch):
    """Fused kernel accumulation vs the pair-list eager path, forward and
    backward, on pins that include a query's own ray, the theta seam and
    several pins per slot."""
    torch.manual_seed(0)
    device = 'cuda'
    num_pins, num_queries, num_slots = 400, 96, 12
    pin_theta = torch.rand(num_pins, device=device) * pins.TWO_PI
    pin_z = torch.rand(num_pins, device=device) * 100.
    eps_theta = torch.rand(num_pins, device=device) * 0.3 + 0.05
    eps_z = torch.rand(num_pins, device=device) * 8. + 2.
    slot = torch.randint(0, num_slots, [num_pins], device=device)
    pin_r = torch.rand(num_pins, device=device) * 200.
    target = torch.rand(num_pins, device=device) * 200.
    theta_q = torch.cat([pin_theta[:16], torch.rand(num_queries - 20, device=device) * pins.TWO_PI,
                         torch.tensor([0., 1e-4, pins.TWO_PI - 1e-4, 3.], device=device)])
    z_q = torch.cat([pin_z[:16], torch.rand(num_queries - 16, device=device) * 100.])
    dr = torch.tensor(16., device=device)
    cells = pins.PinCells(pin_theta, pin_z, eps_theta, eps_z, 0., 100.)
    order = cells.order

    def run():
        leaves = [t.clone().requires_grad_(True) for t in (theta_q, z_q, pin_theta, pin_z, pin_r, target, dr)]
        tq, zq, pt, pz, pr, tg, d = leaves
        mass = torch.zeros([num_queries, num_slots], device=device)
        out = pins._accumulate_anchor_sums(
            tq, zq, cells, pt[order], pz[order], eps_theta[order], eps_z[order], slot[order],
            pr[order], tg[order], d, num_slots, mass, torch.zeros_like(mass), torch.zeros_like(mass))
        # Weight the sums so every branch of the backward is exercised; scale
        # the singular own-ray masses down so the comparison is not dominated
        # by 1e12 kernel values.
        weights = [torch.randn_like(out[0]) for _ in range(3)]
        loss = sum(((o.clamp(max=1e6) * w_).sum() for o, w_ in zip(out, weights)))
        grads = torch.autograd.grad(loss, leaves)
        return [o.detach() for o in out], [g.detach() for g in grads]

    torch.manual_seed(5)
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '1')
    fused_out, fused_grads = run()
    torch.manual_seed(5)
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '0')
    eager_out, eager_grads = run()
    for a, e in zip(fused_out, eager_out):
        torch.testing.assert_close(a, e, atol=1e-3, rtol=1e-5)
    for a, e in zip(fused_grads, eager_grads):
        scale = float(e.abs().max().clamp(min=1.0))
        torch.testing.assert_close(a, e, atol=1e-5 * scale, rtol=1e-4)


def test_sampled_subset_persists_for_rebin_interval():
    """The per-step pin subset is redrawn only when the layout is rebuilt,
    otherwise a fresh subset would force a rebuild every step."""
    model = scene()
    model.cfg['sample_count_pins'] = 20
    model.cfg['model_pin_rebin_interval'] = 3
    torch.manual_seed(0)
    seen = []
    for _ in range(7):
        # Like the training step: the shared leaves carry the sampled pins.
        shared = tuple(x.detach() for x in model.get_shared_transform_tensors())
        model.get_slice_to_spiral_transform(shared=shared)
        seen.append(model._pin_view['indices'].clone())
    assert torch.equal(seen[0], seen[1]) and torch.equal(seen[1], seen[2])
    assert not torch.equal(seen[2], seen[3])
    assert torch.equal(seen[3], seen[4]) and torch.equal(seen[4], seen[5])
    assert not torch.equal(seen[5], seen[6])
    assert model._pin_rebuilds == 3
    # Export still pushes the whole registry, and the next training step
    # draws afresh rather than reusing the full view.
    assert model.compute_pins(full=True).shape[0] == model.pin_registry.num_pins
    model.get_shared_transform_tensors()
    assert model._pin_view['indices'].numel() == 20


def test_step_patch_pin_subset_is_exact_for_chosen_patches():
    """With step patches set, the training subset is every pin of those
    patches plus all chain/isolated pins at registry footprints; the loss's
    patch draw and the pins then agree by construction."""
    model = scene()
    reg = model.pin_registry
    assert reg.patch_index is not None
    patches = torch.unique(reg.patch_index[reg.patch_index >= 0])
    chosen = patches[:1]
    model.cfg['sample_count_pins'] = 10 ** 6
    model.set_step_pin_patches(chosen)
    pins_t = model.compute_pins(subsample=True)
    idx = model._pin_view['indices']
    expected = (reg.kind != pins.PIN_KIND_PATCH) | torch.isin(reg.patch_index, chosen)
    assert torch.equal(torch.sort(idx).values, torch.nonzero(expected, as_tuple=True)[0])
    assert torch.equal(model._pin_view['eps_theta'], reg.eps_theta[idx])
    assert torch.equal(model.active_step_pin_patches(), chosen)
    assert pins_t.shape[0] == idx.numel()
    # Budget thinning keeps every chain pin and widens the remaining patch pins.
    model.cfg['sample_count_pins'] = int((reg.kind != pins.PIN_KIND_PATCH).sum()) + 4
    model.compute_pins(subsample=True)
    idx2 = model._pin_view['indices']
    assert idx2.numel() == model.cfg['sample_count_pins']
    assert bool(torch.isin(torch.nonzero(reg.kind != pins.PIN_KIND_PATCH, as_tuple=True)[0], idx2).all())
    assert (model._pin_view['eps_z'] >= reg.eps_z[idx2]).all()
    model.set_step_pin_patches(None)
    model.compute_pins(subsample=True)
    assert model.active_step_pin_patches() is None
