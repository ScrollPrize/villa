"""Which interpolant a Cartesian flow field integrates, and how fits keep it.

model_flow_field_direct_lr (on by default) asks the Cartesian field to sample
its coarse lattice directly at each point instead of upsampling it to the fine
lattice every step. Only the fused kernels sample directly (Triton on CUDA,
Metal on MPS). Elsewhere, and for lattices a kernel cannot take, a fresh fit
integrates the upsampled field. For the same parameters the two interpolants
place different surfaces, so:

- a checkpoint records the interpolant its fit integrated
  (``flow_field_interpolant``). An older checkpoint maps to what main
  integrated where it was written: an MPS one (it carries
  ``torch_mps_rng_state``) or a CPU one to the upsampled field, a CUDA one to
  direct sampling;
- startup resume, in-session load, model-stage rebuilds, the flattener and
  vesuvius' infer_winding_volume keep that interpolant. They integrate a
  direct one eagerly where no fused kernel runs. It changes only when asked
  (FIT_SPIRAL_DIRECT_LR, or a model_flow_field_direct_lr set otherwise than
  the checkpoint's), and the change is logged;
- the model-state digest (stamped into checkpoints, previews and commits,
  and keying preview reuse) names the interpolant, so it equals no digest
  written before, when one digest covered both surfaces;
- a lattice with a single cell along any axis (the documented 100-slice
  window with no z margin has one coarse z cell) reaches neither fused
  kernel: it falls back to the upsampled field, in training and export,
  unless FIT_SPIRAL_DIRECT_LR=1 asks for direct sampling, which is then
  integrated eagerly and names the shape.

The Metal dispatch is checked on any machine. The ``simulated_metal`` fixture
makes flow_metal accept CPU float32 lattices and run the kernels' arithmetic
in eager PyTorch (direct_lr_reference, which test_flow_direct_fused holds the
real kernels to), after the kernels' own launch-time shape check.
The ``mps`` tests repeat the dispatch checks on a real Apple GPU.
"""
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import checkpoint_io  # noqa: E402
import fit_spiral  # noqa: E402
import flatten_spiral_checkpoint  # noqa: E402
import flow_fields  # noqa: E402
import flow_metal  # noqa: E402
import flow_triton  # noqa: E402
from config import Config  # noqa: E402
from flow_fields import CartesianFlowField  # noqa: E402
from direct_lr_reference import reference_integrate  # noqa: E402
from transforms import SpiralAndTransform  # noqa: E402

MPS = torch.backends.mps.is_available()
needs_mps = pytest.mark.skipif(not MPS, reason='needs an Apple GPU (MPS)')

Z_SPAN = 192   # 12 fine (16-voxel) cells per axis, 2 coarse ones at scale 6
RADIUS = 96
# One surface, in spiral-space voxels: float32 rounding of two summation
# orders (the eager direct integrator against the reference it is checked
# by) stays below 1e-4, two orders under what separates the interpolants here.
SAME = {'rtol': 0, 'atol': 1e-4}
# Metal on MPS against the eager integrator on the CPU: float32 with FMA
# contraction and another summation order through 3 RK4 steps of 2 slabs, about
# 2e-6 of the 192-voxel flow domain, as test_flow_direct_fused allows the
# kernels against the reference.
ACROSS_DEVICES = {'rtol': 0, 'atol': 1e-3}


# --------------------------------------------------------------------------
# Metal's dispatch on the CPU
# --------------------------------------------------------------------------

def _cpu_metal_available(*tensors):
    if os.environ.get('FIT_SPIRAL_METAL', '1') == '0':
        return False
    return all(t.device.type == 'cpu' and t.dtype == torch.float32
               for t in tensors if t is not None)


class _ReferenceDirect(torch.autograd.Function):
    # rk4_direct_integrate's contract on the eager reference: field
    # gradients go, unscaled, into the caller's accumulators.

    @staticmethod
    def forward(ctx, y0, low, high, lo_scale, hi_scale, acc_lo, acc_hi, h,
                n_steps, reverse):
        ctx.save_for_backward(y0, low, high)
        ctx.args = (lo_scale, hi_scale, acc_lo, acc_hi, h, n_steps, reverse)
        with torch.no_grad():
            return reference_integrate(y0, low, high, lo_scale, hi_scale, h,
                                        n_steps, reverse)

    @staticmethod
    def backward(ctx, grad):
        y0, low, high = ctx.saved_tensors
        lo_scale, hi_scale, acc_lo, acc_hi, h, n_steps, reverse = ctx.args
        with torch.enable_grad():
            leaves = [t.detach().requires_grad_(True) for t in (y0, low, high)]
            out = reference_integrate(*leaves, lo_scale, hi_scale, h, n_steps,
                                       reverse)
            g_y, g_lo, g_hi = torch.autograd.grad(out, leaves, grad)
        if acc_lo is not None:
            acc_lo.add_(g_lo / lo_scale)
            acc_hi.add_(g_hi / hi_scale)
        return (g_y,) + (None,) * 9


def _simulated_direct_integrate(y0, low, high, lo_scale, hi_scale, acc_lo,
                                acc_hi, h, n_steps, reverse=False):
    # The kernels' own launch check first: it refuses lattices they cannot take.
    flow_metal._params(y0.shape[0], low, high, lo_scale, hi_scale, h, n_steps)
    if torch.is_grad_enabled() and y0.requires_grad:
        return _ReferenceDirect.apply(y0, low, high, lo_scale, hi_scale,
                                      acc_lo, acc_hi, h, n_steps, reverse)
    with torch.no_grad():
        return reference_integrate(y0, low, high, lo_scale, hi_scale, h,
                                    n_steps, reverse)


@pytest.fixture
def simulated_metal(monkeypatch):
    monkeypatch.delenv('FIT_SPIRAL_DIRECT_LR', raising=False)
    monkeypatch.delenv('FIT_SPIRAL_METAL', raising=False)
    monkeypatch.setattr(flow_metal, 'rk4_metal_available', _cpu_metal_available)
    monkeypatch.setattr(flow_metal, 'rk4_direct_integrate',
                        _simulated_direct_integrate)


@pytest.fixture
def no_overrides(monkeypatch):
    monkeypatch.delenv('FIT_SPIRAL_DIRECT_LR', raising=False)
    monkeypatch.delenv('FIT_SPIRAL_METAL', raising=False)


# --------------------------------------------------------------------------
# A small fitted model and its checkpoints
# --------------------------------------------------------------------------

def _config(**overrides):
    cfg = Config().as_dict()
    cfg.update(model_flow_field_type='cartesian',
               model_flow_field_direct_lr=True,
               model_gap_expander_num_windings=10,
               model_gap_expander_capacity_windings=10,
               model_flow_bounds_radius=RADIUS,
               model_flow_bounds_z_margin=0)
    cfg.update(overrides)
    return cfg


def _model(cfg, z_begin=0, z_end=Z_SPAN, seed=23, device='cpu'):
    margin = int(cfg['model_flow_bounds_z_margin'])
    radius = int(cfg['model_flow_bounds_radius'])
    zs = torch.arange(z_begin, z_end, dtype=torch.float32)
    umbilicus = torch.stack(
        [zs, torch.full_like(zs, 3.0), torch.full_like(zs, -2.0)], dim=-1)
    model = SpiralAndTransform(
        flow_integration_steps=int(cfg['model_num_flow_integration_steps']),
        flow_integration_solver=str(cfg['model_flow_integration_solver']),
        flow_min_corner_zyx=torch.tensor([z_begin - margin, -radius, -radius],
                                         device=device),
        flow_max_corner_zyx=torch.tensor([z_end + margin, radius, radius],
                                         device=device),
        umbilicus_zyx=umbilicus.to(device),
        config=cfg, spiral_outward_sense='CW').to(device)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel() > 1:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.01)
        # Coarse-lattice detail, which the two interpolants place differently.
        coarse = model.flow_field.flows[0]
        coarse.copy_(torch.randn(coarse.shape, generator=generator) * 0.05)
    return model


def _like(cfg, state, **kwargs):
    model = _model(cfg, **kwargs)
    model.load_state_dict(state)
    return model


def _points(n=257, z_begin=0, z_end=Z_SPAN, seed=5):
    generator = torch.Generator().manual_seed(seed)
    z = torch.rand(n, generator=generator) * (z_end - z_begin - 8) + z_begin + 4
    theta = torch.rand(n, generator=generator) * (2 * math.pi)
    radius = torch.rand(n, generator=generator) * 60 + 20
    return torch.stack([z, 3.0 + torch.sin(theta) * radius,
                        -2.0 + torch.cos(theta) * radius], dim=-1)


def _surface(model, points):
    device = next(model.parameters()).device
    with torch.no_grad():
        return model.get_slice_to_spiral_transform()(points.to(device)).cpu()


def _references(state, points):
    """The surfaces `state` places under each interpolant, without pinning:
    upsampled from a config that does not ask for direct sampling, direct from
    one that does, where a fused (here simulated) kernel runs."""
    upsampled = _surface(_like(_config(model_flow_field_direct_lr=False), state), points)
    direct = _surface(_like(_config(), state), points)
    # The two interpolants place these parameters measurably apart.
    assert float((direct - upsampled).abs().max()) > 1e-2
    return upsampled, direct


def _umbilicus_file(tmp_path, z_begin=0, z_end=Z_SPAN):
    path = tmp_path / 'umbilicus.json'
    path.write_text(json.dumps({'control_points': [
        {'z': z_begin, 'y': 3.0, 'x': -2.0}, {'z': z_end, 'y': 3.0, 'x': -2.0}]}))
    return path


def _legacy(state, cfg, *, mps=False, cuda=False):
    """A checkpoint as main wrote it: no recorded interpolant; an MPS fit adds
    its RNG state, and a CUDA machine one state per device."""
    payload = {'schema_version': 2, 'cfg': dict(cfg),
               'spiral_and_transform': state, 'z_begin': 0, 'z_end': Z_SPAN,
               'spiral_outward_sense': 'CW',
               'torch_cuda_rng_states': ([torch.zeros(16, dtype=torch.uint8)]
                                         if cuda else [])}
    if mps:
        payload['torch_mps_rng_state'] = torch.zeros(16, dtype=torch.uint8)
    return payload


def _flattened(checkpoint, tmp_path):
    # The flattener's model loader (its export runs this model).
    config = flatten_spiral_checkpoint._checkpoint_config(checkpoint)
    return flatten_spiral_checkpoint._build_model(
        checkpoint, config, _umbilicus_file(tmp_path), torch.device('cpu'))


class _Fit(fit_spiral.FitContext):
    """A FitContext holding a real model, optimiser and schedule, and nothing
    else: enough to write a checkpoint and to load one in session."""

    def __init__(self, model, cfg):  # noqa: D107 - skips FitContext.__init__
        self.config = {**cfg, 'z_begin': 0, 'z_end': Z_SPAN}
        self.spiral_and_transform = model
        self.gap_expander_params = [model.gap_expander_params.logits]
        self.optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimiser, gamma=0.5)
        self.start_iteration = 0
        self.device = torch.device('cpu')
        self.model_z_begin, self.model_z_end = 0, Z_SPAN
        self.lasagna_scale = None
        self.lasagna_source = None
        self.base_shape_zyx = None
        self.winding_inference = None
        self.spiral_outward_sense = 'CW'
        self.z_direction_is_top_to_bottom = False
        self.interactive_driver = None
        self.paths = SimpleNamespace(volume_uuid=None)
        self.winding_model_mode = False

    def _checkpoint_input_manifest(self):
        return {}


# --------------------------------------------------------------------------
# Checkpoint identity: reconstruction keeps the interpolant a fit integrated
# --------------------------------------------------------------------------

@pytest.mark.parametrize('origin,expected', [
    ('mps', 'upsampled'), ('cpu', 'upsampled'), ('cuda', 'direct')])
def test_the_flattener_integrates_a_legacy_checkpoints_interpolant(
        simulated_metal, tmp_path, origin, expected):
    # Main fitted model_flow_field_direct_lr directly only on CUDA (Triton);
    # an MPS or CPU fit integrated the upsampled field. The flattener runs on
    # an Apple GPU here (simulated), where Metal samples directly.
    fitted = _model(_config())
    points = _points()
    references = dict(zip(('upsampled', 'direct'),
                          _references(fitted.state_dict(), points)))
    checkpoint = _legacy(fitted.state_dict(), _config(),
                         mps=origin == 'mps', cuda=origin == 'cuda')
    torch.testing.assert_close(_surface(_flattened(checkpoint, tmp_path), points),
                               references[expected], **SAME)


def test_a_legacy_cuda_checkpoint_keeps_direct_sampling_without_a_fused_kernel(
        no_overrides, tmp_path):
    # On the CPU no fused kernel runs; a checkpoint fitted with direct
    # sampling is integrated eagerly instead of on the upsampled field.
    fitted = _model(_config())
    points = _points()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(flow_metal, 'rk4_metal_available', _cpu_metal_available)
        patch.setattr(flow_metal, 'rk4_direct_integrate',
                      _simulated_direct_integrate)
        _, direct = _references(fitted.state_dict(), points)
    rebuilt = _flattened(_legacy(fitted.state_dict(), _config(), cuda=True), tmp_path)
    torch.testing.assert_close(_surface(rebuilt, points), direct, **SAME)


def test_a_checkpoint_saved_with_metal_off_keeps_its_field_with_metal_on(
        simulated_metal, monkeypatch, tmp_path):
    # The reviewer's case: the same checkpoint saved where no kernel ran
    # (FIT_SPIRAL_METAL=0), then reconstructed with Metal on.
    cfg = _config()
    fitted = _model(cfg)
    points = _points()
    monkeypatch.setenv('FIT_SPIRAL_METAL', '0')
    integrated = _surface(fitted, points)
    off = _Fit(fitted, cfg)._checkpoint_payload(5)
    monkeypatch.delenv('FIT_SPIRAL_METAL')
    on = _Fit(fitted, cfg)._checkpoint_payload(5)

    torch.testing.assert_close(_surface(_flattened(off, tmp_path), points),
                               integrated, **SAME)
    # Identical parameters, different surfaces: different digests, so a
    # preview of one is never re-shown for the other.
    assert off['model_state_sha256'] != on['model_state_sha256']
    assert off['flow_field_interpolant'] == 'upsampled'
    assert on['flow_field_interpolant'] == 'direct_lr'


def test_an_in_session_load_integrates_the_checkpoints_interpolant(simulated_metal):
    cfg = _config()
    fitted = _model(cfg)
    points = _points()
    upsampled, _ = _references(fitted.state_dict(), points)
    session = _Fit(_model(cfg, seed=99), cfg)   # a fresh session: Metal, direct
    checkpoint = session._checkpoint_payload(7)
    checkpoint.pop('flow_field_interpolant', None)
    checkpoint.update(spiral_and_transform=fitted.state_dict(),
                      torch_mps_rng_state=torch.zeros(16, dtype=torch.uint8))

    session.apply_checkpoint(checkpoint)

    torch.testing.assert_close(_surface(session.spiral_and_transform, points),
                               upsampled, **SAME)
    assert session._checkpoint_payload(8)['flow_field_interpolant'] == 'upsampled'
    assert session.model_state_digest() == checkpoint_io.model_state_sha256(
        fitted.state_dict(), (), 0, Z_SPAN, flow_interpolant='upsampled')


def test_a_startup_resume_integrates_the_checkpoints_interpolant(
        simulated_metal, cpu, store_path, tmp_path):
    # The command line's resume, through the real model build on the CPU.
    pytest.importorskip('vc_object_store')
    from test_object_store_startup import _context
    trained = _context(store_path, tmp_path)
    trained.load_host_inputs()
    trained.build_device_state()
    model = trained.spiral_and_transform
    with torch.no_grad():
        coarse = model.flow_field.flows[0]
        coarse.copy_(torch.randn(coarse.shape, generator=torch.Generator().manual_seed(3)) * 0.05)
    checkpoint = trained._checkpoint_payload(1)
    checkpoint.pop('flow_field_interpolant', None)       # as main wrote it on MPS
    checkpoint['torch_mps_rng_state'] = torch.zeros(16, dtype=torch.uint8)
    path = tmp_path / 'legacy_mps.ckpt'
    torch.save(checkpoint, path)
    points = _points(z_begin=8, z_end=56)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('FIT_SPIRAL_METAL', '0')
        upsampled = _surface(model, points)

    resumed = _context(store_path, tmp_path, resume_path=str(path))
    resumed.load_host_inputs()
    resumed.build_device_state()

    assert resumed.start_iteration == 1
    torch.testing.assert_close(_surface(resumed.spiral_and_transform, points),
                               upsampled, **SAME)


# --------------------------------------------------------------------------
# Switching, logging, and the digest
# --------------------------------------------------------------------------

def test_only_fit_spiral_direct_lr_switches_a_checkpoints_interpolant(
        simulated_metal, monkeypatch, tmp_path, capsys):
    fitted = _model(_config())
    points = _points()
    upsampled, direct = _references(fitted.state_dict(), points)
    legacy = _legacy(fitted.state_dict(), _config(), mps=True)
    capsys.readouterr()
    monkeypatch.setenv('FIT_SPIRAL_DIRECT_LR', '1')
    switched = _flattened(legacy, tmp_path)
    torch.testing.assert_close(_surface(switched, points), direct, **SAME)
    out = capsys.readouterr().out
    assert 'FIT_SPIRAL_DIRECT_LR=1' in out and 'upsampled' in out
    assert switched.flow_interpolant() == 'direct_lr'
    # A fit run with FIT_SPIRAL_DIRECT_LR=0 records the upsampled field, and
    # its checkpoint keeps it once the override is gone.
    monkeypatch.setenv('FIT_SPIRAL_DIRECT_LR', '0')
    recorded = _Fit(fitted, _config())._checkpoint_payload(5)
    monkeypatch.delenv('FIT_SPIRAL_DIRECT_LR')
    assert recorded['flow_field_interpolant'] == 'upsampled'
    torch.testing.assert_close(_surface(_flattened(recorded, tmp_path), points),
                               upsampled, **SAME)


def test_a_fresh_field_reports_the_interpolant_it_integrates(simulated_metal, monkeypatch):
    field = _model(_config()).flow_field
    assert field.effective_interpolant() == 'direct_lr'
    monkeypatch.setenv('FIT_SPIRAL_METAL', '0')
    assert field.effective_interpolant() == 'upsampled'
    monkeypatch.delenv('FIT_SPIRAL_METAL')
    monkeypatch.setenv('FIT_SPIRAL_DIRECT_LR', '0')
    assert field.effective_interpolant() == 'upsampled'
    monkeypatch.delenv('FIT_SPIRAL_DIRECT_LR')
    assert _model(_config(model_flow_field_direct_lr=False)).flow_field \
        .effective_interpolant() == 'upsampled'
    # Field types with a single interpolant record none.
    assert _model(_config(model_flow_field_type='cylindrical')).flow_interpolant() is None


def test_the_digest_names_either_interpolant():
    # Before the interpolant was recorded, one digest covered both surfaces of
    # the same parameters: CUDA stamped it on directly sampled ones, MPS and
    # the CPU on upsampled ones. A digest naming either interpolant therefore
    # equals no older one.
    state = _model(_config()).state_dict()
    unrecorded = checkpoint_io.model_state_sha256(state, (), 0, Z_SPAN)
    direct = checkpoint_io.model_state_sha256(
        state, (), 0, Z_SPAN, flow_interpolant='direct_lr')
    upsampled = checkpoint_io.model_state_sha256(
        state, (), 0, Z_SPAN, flow_interpolant='upsampled')
    assert len({unrecorded, direct, upsampled}) == 3


def test_a_cuda_checkpoint_loaded_on_the_cpu_is_never_main_s_surface_digest(no_overrides):
    # The reviewer's case: main loaded a CUDA checkpoint in session on the
    # CPU, integrated the upsampled field and reported the unrecorded digest
    # (which its published previews and commits carry). This session
    # integrates the checkpoint's direct sampling, another surface, so it must
    # report another digest.
    cfg = _config()
    fitted = _model(cfg)
    session = _Fit(_model(cfg, seed=99), cfg)
    checkpoint = session._checkpoint_payload(7)
    checkpoint.pop('flow_field_interpolant', None)
    checkpoint.update(spiral_and_transform=fitted.state_dict(),
                      torch_cuda_rng_states=[torch.zeros(16, dtype=torch.uint8)])
    mains = checkpoint_io.model_state_sha256(fitted.state_dict(), (), 0, Z_SPAN)

    session.apply_checkpoint(checkpoint)

    assert session.spiral_and_transform.flow_interpolant() == 'direct_lr'
    assert session.model_state_digest() != mains


def test_an_unknown_recorded_interpolant_is_refused():
    with pytest.raises(ValueError, match='unknown flow interpolant'):
        checkpoint_io.checkpoint_flow_interpolant(
            {'cfg': _config(), 'flow_field_interpolant': 'bicubic'})


def test_a_changed_direct_lr_setting_decides_only_a_resume(simulated_metal, capsys):
    cfg = _config()
    fitted = _model(cfg)
    points = _points()
    upsampled, direct = _references(fitted.state_dict(), points)
    checkpoint = _Fit(fitted, cfg)._checkpoint_payload(7)
    assert checkpoint['flow_field_interpolant'] == 'direct_lr'
    off = _config(model_flow_field_direct_lr=False)
    session = _Fit(_model(off, seed=99), off)

    session.apply_checkpoint(checkpoint)
    torch.testing.assert_close(_surface(session.spiral_and_transform, points),
                               direct, **SAME)

    capsys.readouterr()
    session.apply_checkpoint(checkpoint, config_decides=True)
    torch.testing.assert_close(_surface(session.spiral_and_transform, points),
                               upsampled, **SAME)
    out = capsys.readouterr().out
    assert out.count('WARNING: checkpoint flow field: model_flow_field_direct_lr is False '
                     'here and was True') == 1
    assert 'the fitted surface moves' in out


def test_every_load_of_a_legacy_cuda_checkpoint_says_it_is_inferred(simulated_metal, capsys):
    cfg = _config()
    session = _Fit(_model(cfg), cfg)
    checkpoint = session._checkpoint_payload(7)
    checkpoint.pop('flow_field_interpolant')
    checkpoint['torch_cuda_rng_states'] = [torch.zeros(16, dtype=torch.uint8)]
    capsys.readouterr()

    session.apply_checkpoint(checkpoint)
    session.apply_checkpoint(checkpoint)

    assert capsys.readouterr().out.count('direct_lr (inferred, not recorded') == 2


def test_a_checkpoint_of_another_field_type_is_left_to_the_preflight():
    # A Cartesian checkpoint resumed into a fit configured with another field
    # type: keeping its interpolant must not raise before the preflight
    # reports the model-shaping mismatch.
    cylindrical = _config(model_flow_field_type='cylindrical')
    session = _Fit(_model(cylindrical), cylindrical)
    checkpoint = _legacy(_model(_config()).state_dict(), _config(), mps=True)
    session._keep_checkpoint_interpolant(checkpoint)
    assert session.spiral_and_transform.flow_interpolant() is None
    verdict = session.inspect_checkpoint(checkpoint)
    assert not verdict.accepted
    assert 'model_flow_field_type' in verdict.message()


def test_a_one_cell_lattice_never_reaches_triton(monkeypatch):
    # Triton's backward clamps its lower corner to size - 2 as Metal's does,
    # which is -1 on a single-cell axis: it would read and add outside the
    # lattice. Such lattices take the path they take without a kernel.
    monkeypatch.delenv('FIT_SPIRAL_DIRECT_LR', raising=False)
    monkeypatch.setattr(flow_triton, 'rk4_triton_available',
                        lambda *tensors: True)
    thin = _thin(0)
    backend, reason, shape = flow_fields._fused_direct_backend(thin.flows[0], thin.flows[1])
    assert backend is None and shape and '1 cell along z' in reason
    assert thin.effective_interpolant() == 'upsampled'
    monkeypatch.setenv('FIT_SPIRAL_DIRECT_LR', '1')
    assert thin._resolve_interpolant(log=False) == ('direct_lr', flow_fields._EagerDirect)
    wide = CartesianFlowField([12, 12, 12], spatial_scale_factor=6, direct_lr=True)
    assert flow_fields._fused_direct_backend(wide.flows[0], wide.flows[1])[0] is flow_triton


def test_the_winding_volume_seeds_integrate_the_checkpoint_s_interpolant(no_overrides, tmp_path):
    # vesuvius' infer_winding_volume rebuilds the fitted model to place its
    # seed rays: a CUDA checkpoint keeps its direct sampling on the CPU too.
    import importlib.util
    path = (Path(__file__).resolve().parents[2] / 'vesuvius/src/vesuvius/neural_tracing'
            / 'winding_models/infer_winding_volume.py')
    spec = importlib.util.spec_from_file_location('infer_winding_volume_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = _model(_config()).state_dict()
    umbilicus = _umbilicus_file(tmp_path)
    seeds = {}
    for name, extra in (('cuda', {}), ('direct', {'flow_field_interpolant': 'direct_lr'}),
                        ('upsampled', {'flow_field_interpolant': 'upsampled'})):
        checkpoint = {**_legacy(state, _config(), cuda=True), **extra}
        torch.save(checkpoint, tmp_path / f'{name}.ckpt')
        args = SimpleNamespace(fit_checkpoint=str(tmp_path / f'{name}.ckpt'),
                               umbilicus=str(umbilicus), z_range=None,
                               winding_range=(1, 3), winding_step=1, seed_spacing=8)
        seeds[name] = module.build_seed_rays(args, device='cpu')['seed_xyz']
    np = pytest.importorskip('numpy')
    assert seeds['cuda'].shape == seeds['direct'].shape
    np.testing.assert_allclose(seeds['cuda'], seeds['direct'], rtol=0, atol=1e-4)
    assert seeds['cuda'].shape != seeds['upsampled'].shape or \
        float(np.abs(seeds['cuda'] - seeds['upsampled']).max()) > 1e-3


# --------------------------------------------------------------------------
# One-cell lattice axes
# --------------------------------------------------------------------------

def _thin(axis, **kwargs):
    # Fine lattice 6 cells long on `axis` (12 elsewhere), coarse at scale 6:
    # one coarse cell on that axis, as the documented 100-slice window with no
    # z margin gives along z.
    resolution = [12, 12, 12]
    resolution[axis] = 6
    torch.manual_seed(11)
    field = CartesianFlowField(resolution, spatial_scale_factor=6,
                               num_stages=2, direct_lr=True, **kwargs)
    with torch.no_grad():
        field.flows[0].uniform_(-0.03, 0.03)
        field.flows[1].uniform_(-0.012, 0.012)
    return field


def _train_and_export(field, device='cpu'):
    field = field.to(device)
    points = torch.rand(64, 3, generator=torch.Generator().manual_seed(2)).to(device)
    y = points.clone().requires_grad_(True)
    out = field.get_integrator()(y, 1.0 / 3, 3)
    out.square().sum().backward()
    field.apply_accumulated_field_grad()
    with torch.no_grad():
        exported = field.get_integrator()(points, 1.0 / 3, 3)
    for tensor in (out, y.grad, field.flows[0].grad, field.flows[1].grad, exported):
        assert torch.isfinite(tensor).all()
    return out.detach().cpu(), exported.cpu()


@pytest.mark.parametrize('axis', [0, 1, 2], ids=['z', 'y', 'x'])
def test_a_one_cell_coarse_axis_falls_back_to_the_upsampled_field(
        simulated_metal, axis, monkeypatch):
    trained, exported = _train_and_export(_thin(axis))
    assert _thin(axis).effective_interpolant() == 'upsampled'
    monkeypatch.setenv('FIT_SPIRAL_METAL', '0')
    eager, eager_exported = _train_and_export(_thin(axis))
    torch.testing.assert_close(trained, eager, rtol=0, atol=0)
    torch.testing.assert_close(exported, eager_exported, rtol=0, atol=0)


@pytest.mark.parametrize('axis', [0, 1, 2], ids=['z', 'y', 'x'])
def test_an_explicit_direct_request_on_a_one_cell_lattice_integrates_eagerly(
        simulated_metal, monkeypatch, capsys, axis):
    # The way back to direct sampling where the kernels do not take the
    # lattice: FIT_SPIRAL_DIRECT_LR=1 integrates it eagerly, says why, and
    # places the reference integrator's points (one-cell axes included).
    monkeypatch.setenv('FIT_SPIRAL_DIRECT_LR', '1')
    field = _thin(axis)
    assert field.effective_interpolant() == 'direct_lr'
    capsys.readouterr()
    trained, exported = _train_and_export(field)
    assert '1 cell along ' + 'zyx'[axis] in capsys.readouterr().out
    points = torch.rand(64, 3, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        expected = reference_integrate(points, field.flows[0].detach(),
                                       field.flows[1].detach(), 1.0, 1.0,
                                       1.0 / 3, 3, False)
    torch.testing.assert_close(exported, expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(trained, expected, rtol=0, atol=1e-6)


def test_the_documented_narrow_window_trains_and_exports(simulated_metal):
    # README "fit window": 100 slices with model_flow_bounds_z_margin 0 give
    # six fine z cells and one coarse one at the default 16-voxel spacing.
    cfg = _config(model_flow_bounds_radius=3200)
    model = _model(cfg, z_begin=0, z_end=100)
    assert model.flow_field.flows[0].shape[2] == 1
    points = _points(z_begin=0, z_end=100)
    spiral = model.get_slice_to_spiral_transform()(points)
    spiral.square().mean().backward()
    model.flow_field.apply_accumulated_field_grad()
    assert torch.isfinite(model.flow_field.flows[0].grad).all()
    assert torch.isfinite(_surface(model, points)).all()
    assert model.flow_interpolant() == 'upsampled'


# --------------------------------------------------------------------------
# The same on a real Apple GPU
# --------------------------------------------------------------------------

@needs_mps
@pytest.mark.parametrize('axis', [0, 1, 2], ids=['z', 'y', 'x'])
def test_mps_one_cell_axis_falls_back_in_training_and_export(no_overrides, axis):
    trained, exported = _train_and_export(_thin(axis), 'mps')
    assert _thin(axis).to('mps').effective_interpolant() == 'upsampled'
    assert torch.isfinite(trained).all() and torch.isfinite(exported).all()


@needs_mps
def test_mps_flattener_keeps_a_legacy_mps_checkpoints_upsampled_field(
        no_overrides, tmp_path):
    cfg = _config()
    fitted = _model(cfg)
    points = _points()
    upsampled = _surface(_like(_config(model_flow_field_direct_lr=False),
                               fitted.state_dict(), device='mps'), points)
    checkpoint = _legacy(fitted.state_dict(), cfg, mps=True)
    config = flatten_spiral_checkpoint._checkpoint_config(checkpoint)
    rebuilt = flatten_spiral_checkpoint._build_model(
        checkpoint, config, _umbilicus_file(tmp_path), torch.device('mps'))
    torch.testing.assert_close(_surface(rebuilt, points), upsampled, **SAME)


@needs_mps
def test_mps_metal_and_eager_direct_sampling_place_one_surface(no_overrides, tmp_path):
    # The same direct-LR checkpoint through Metal on MPS and through the eager
    # integrator on the CPU: one surface, to float32 rounding.
    cfg = _config()
    fitted = _model(cfg)
    points = _points()
    checkpoint = _legacy(fitted.state_dict(), cfg, cuda=True)
    config = flatten_spiral_checkpoint._checkpoint_config(checkpoint)
    on_mps = flatten_spiral_checkpoint._build_model(
        checkpoint, config, _umbilicus_file(tmp_path), torch.device('mps'))
    on_cpu = _flattened(checkpoint, tmp_path)
    assert on_mps.flow_interpolant() == on_cpu.flow_interpolant() == 'direct_lr'
    torch.testing.assert_close(_surface(on_mps, points), _surface(on_cpu, points),
                               **ACROSS_DEVICES)
