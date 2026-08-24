"""Constraint baking for transform resets ("plan B").

At each reset the whole live transform is frozen: its exact state is
snapshotted to CPU, every scroll-space training input is pushed through the
frozen inverse into (near-)canonical spiral space, and the live parameters
restart from the identity (dr_per_winding preserved). Nothing frozen runs
during training afterwards; the frozen chain is materialised from the CPU
snapshots only for export/metrics, where the final map is
``F_1 o F_2 o ... o F_live`` (earliest epoch nearest scroll space).

Space bookkeeping: with ``S_k`` the epoch-k scroll(-of-that-epoch)->spiral
transform, baked coordinates satisfy ``b_k = S_k(b_{k-1})`` (``b_0`` is true
scroll space), so the composed scroll->spiral map is
``ComposeTransform([S_1, ..., S_k, S_live])``.

Two frame consequences of baking that are deliberate, not incidental:

- Baked inputs are z-axis-centred (the frozen umbilicus stage removed the
  shear), so post-reset epochs use an identity umbilicus stage; any residual
  umbilicus error of the frozen epoch is committed into the constraints.
- Baked inputs are sense-normalised (the frozen chain's optional x-flip is
  already applied), so post-reset epochs always run with outward sense 'CW'.

Inversion error is permanent and compounding: each bake rewrites the
constraints through the RK4 inverse of the fitted forward flow, so callers
must probe and log the round-trip error at every bake (probe_round_trip_error)
and keep bake-time integration settings identical to the training values
(snapshots record them for exactly that reason).
"""

import torch
import pyro.distributions

from transforms import SpiralAndTransform, UmbilicusTransform


# Fixed, rank-independent chunking keeps a bake deterministic and identical
# across DDP ranks (parameters are identical post-allreduce, and baking
# consumes no RNG).
BAKE_CHUNK_SIZE = 1 << 20

# The configuration keys SpiralAndTransform's constructor reads. They are
# stored in each snapshot so a frozen epoch can be rebuilt exactly even if a
# later (model-stage) rebuild changes the live values.
MODEL_CONSTRUCTION_CONFIG_KEYS = (
    'model_initial_dr_per_winding',
    'model_flow_voxel_resolution',
    'model_flow_field_type',
    'model_num_flow_timesteps',
    'model_num_flow_stages',
    'model_flow_field_direct_lr',
    'model_gap_expander_logit_resolution',
    'model_gap_expander_num_windings',
    'model_gap_expander_lr_scale',
    'model_linear_z_resolution',
)


def identity_umbilicus_like(umbilicus_zyx):
    """An umbilicus table with the same z samples and yx pinned to the axis."""
    zs = torch.as_tensor(umbilicus_zyx)[..., :1]
    return torch.cat([zs, torch.zeros_like(zs), torch.zeros_like(zs)], dim=-1)


def snapshot_frozen_epoch(model, umbilicus_zyx, config, *, probe_error=None):
    """Exact CPU snapshot of the live transform at a reset boundary.

    Captures the full model state_dict, the raw umbilicus table the model's
    umbilicus stage was built from (UmbilicusTransform state is not in the
    state_dict), and every construction argument, so materialize_frozen_model
    can rebuild a transform whose forward/inverse arithmetic is identical to
    the one that produced the baked constraints.
    """
    if model.flow_integration_solver != 'rk4':
        raise ValueError('constraint baking supports only the rk4 solver')
    return {
        'spiral_and_transform': {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        'umbilicus_zyx': torch.as_tensor(
            umbilicus_zyx).detach().cpu().clone().to(torch.float32),
        'spiral_outward_sense': model.spiral_outward_sense,
        'flow_integration_steps': int(model.flow_integration_steps),
        'flow_integration_solver': str(model.flow_integration_solver),
        'flow_min_corner_zyx': model.flow_min_corner_zyx.detach().cpu().clone(),
        'flow_max_corner_zyx': model.flow_max_corner_zyx.detach().cpu().clone(),
        'model_config': {
            key: config[key] for key in MODEL_CONSTRUCTION_CONFIG_KEYS},
        'dr_per_winding': float(
            model.get_dr_per_winding().detach().cpu()),
        'probe_round_trip_error': probe_error,
    }


def materialize_frozen_model(snapshot, config, device):
    """Rebuild one frozen epoch's SpiralAndTransform from its CPU snapshot.

    The snapshot's own construction configuration wins over the live
    configuration for every key it recorded, so the rebuilt transform is the
    fitted ground truth of that epoch regardless of later model-stage edits.
    """
    device = torch.device(device)
    frozen_config = dict(config)
    frozen_config.update(snapshot['model_config'])
    model = SpiralAndTransform(
        flow_integration_steps=snapshot['flow_integration_steps'],
        flow_integration_solver=snapshot['flow_integration_solver'],
        flow_min_corner_zyx=snapshot['flow_min_corner_zyx'].to(device),
        flow_max_corner_zyx=snapshot['flow_max_corner_zyx'].to(device),
        umbilicus_zyx=snapshot['umbilicus_zyx'].to(device),
        config=frozen_config,
        spiral_outward_sense=snapshot['spiral_outward_sense'],
    )
    model.to(device)
    model.load_state_dict(snapshot['spiral_and_transform'])
    model.requires_grad_(False)
    return model


def bake_points(transform, points, *, device=None, chunk_size=BAKE_CHUNK_SIZE):
    """Push ``points`` (..., 3) through ``transform`` chunked, under no_grad.

    Returns a new tensor with the input's shape, dtype and device; the input
    is never mutated. The chunk size is fixed so the evaluation is
    deterministic and identical on every DDP rank.
    """
    source = torch.as_tensor(points)
    flat = source.reshape(-1, 3)
    out = torch.empty_like(flat)
    if flat.numel() == 0:
        return out.view(source.shape)
    with torch.no_grad():
        for start in range(0, len(flat), chunk_size):
            chunk = flat[start:start + chunk_size]
            staged = chunk.to(
                device=device if device is not None else chunk.device,
                dtype=torch.float32)
            baked = transform(staged)
            out[start:start + chunk_size] = baked.to(
                device=out.device, dtype=out.dtype)
    return out.view(source.shape)


def forward_transform(model):
    """The canonical->scroll forward chain of a (frozen) model."""
    return model.get_slice_to_spiral_transform().inv


def _probe_points(model, num_z=6, num_theta=8, num_radii=5):
    """Fixed, RNG-free scroll-space probe set: umbilicus-centred rings
    spanning the fit's z window and the first tens of windings — plausible
    constraint locations."""
    device = model.device
    z_lo = float(model.flow_min_corner_zyx[0])
    z_hi = float(model.flow_max_corner_zyx[0])
    span = z_hi - z_lo
    zs = torch.linspace(
        z_lo + 0.1 * span, z_hi - 0.1 * span, num_z, device=device)
    with torch.no_grad():
        dr = float(model.get_dr_per_winding().detach())
    radius_hi = min(float(model.flow_max_corner_zyx[1]) * 0.5, dr * 40.)
    radii = torch.linspace(
        min(2. * dr, radius_hi), radius_hi, num_radii, device=device)
    thetas = torch.arange(num_theta, device=device) * (
        2. * torch.pi / num_theta)
    z, r, t = torch.meshgrid(zs, radii, thetas, indexing='ij')
    rings = torch.stack(
        [z, r * torch.sin(t), r * torch.cos(t)], dim=-1).reshape(-1, 3)
    with torch.no_grad():
        # Shift the rings onto the umbilicus so they sit where constraints do.
        return model.umbilicus_transform._call(rings)


def probe_round_trip_error(model, *, num_z=6, num_theta=8, num_radii=5):
    """Round-trip error ``|S.inv(S(p)) - p|`` on the fixed probe set.

    Measures the RK4 forward/inverse inconsistency this epoch would commit
    permanently into the constraints; callers log it per bake and watch its
    accumulation across resets.
    """
    scroll = _probe_points(model, num_z, num_theta, num_radii)
    with torch.no_grad():
        slice_to_spiral = model.get_slice_to_spiral_transform()
        spiral = slice_to_spiral(scroll)
        back = slice_to_spiral.inv(spiral)
        error = torch.linalg.norm(back - scroll, dim=-1)
    return {
        'max': float(error.max()),
        'mean': float(error.mean()),
        'num_points': int(error.numel()),
    }


def probe_stretch_factor(model, *, epsilon=2.0,
                         num_z=6, num_theta=8, num_radii=5):
    """Median local stretch of this epoch's scroll->spiral map.

    Finite differences along the three axes at every probe point:
    ``||S(p + eps*e) - S(p)|| / eps``, reduced by the median over all points
    and directions. Baked residuals are re-expressed in units shrunk (or
    grown) by exactly this local stretch, so the accumulated product across
    epochs is the blunt global correction for thresholds that were tuned in
    pre-bake units. It deliberately ignores anisotropy and spatial variation
    — that is what makes it blunt.
    """
    scroll = _probe_points(model, num_z, num_theta, num_radii)
    with torch.no_grad():
        slice_to_spiral = model.get_slice_to_spiral_transform()
        base = slice_to_spiral(scroll)
        ratios = []
        for axis in range(3):
            offset = torch.zeros(3, device=scroll.device)
            offset[axis] = float(epsilon)
            moved = slice_to_spiral(scroll + offset)
            ratios.append(
                torch.linalg.norm(moved - base, dim=-1) / float(epsilon))
        return float(torch.cat(ratios).median())


def reset_parameters(model):
    """The live parameters a reset zeroes (dr_per_winding_logit is kept)."""
    params = []
    for flow_field in model.flow_fields:
        params.extend(flow_field.flows)
    params.append(model.linear_logits)
    params.append(model.gap_expander_params.logits)
    return params


def set_canonical_frame_(model):
    """Point the model at the canonical (post-bake) frame.

    Baked inputs are already z-axis-centred and sense-normalised, so the
    umbilicus stage becomes the identity and the optional x-flip is dropped
    (outward sense 'CW'). Called on every model that trains against baked
    inputs — after a live reset, and after a checkpoint-resume bake replay.
    """
    model.spiral_outward_sense = 'CW'
    identity = identity_umbilicus_like(
        torch.cat([
            model.umbilicus_transform._z,
            model.umbilicus_transform._yx,
        ], dim=-1))
    model.umbilicus_transform = UmbilicusTransform(
        identity.to(model.umbilicus_transform._z.device))


def reset_live_transform_(model):
    """Zero the live transform back to the identity over baked inputs.

    Flow fields, linear logits and gap logits go to zero; the umbilicus stage
    becomes the identity and the sense flip is dropped (set_canonical_frame_).
    ``dr_per_winding_logit`` is deliberately preserved: baked radii encode
    windings at the current dr, and zero gap logits reproduce that spacing
    exactly, while a reinitialised dr would shift every residual.
    """
    with torch.no_grad():
        for parameter in reset_parameters(model):
            parameter.zero_()
    set_canonical_frame_(model)


def clear_optimizer_state_(optimiser, parameters):
    """Drop optimiser moments for reset parameters.

    Stale Adam moments would push the freshly zeroed parameters on the first
    post-reset step, breaking the exact pre/post-bake residual match.
    """
    for parameter in parameters:
        optimiser.state.pop(parameter, None)


def freeze_dr_(model, optimiser=None):
    """Pin dr_per_winding once constraints have been baked.

    Baked radii encode windings at bake-time dr, so any later dr change
    moves the canonical target under every baked constraint at once and the
    gap logits must chase it globally (observed as a monotonic dr drift
    across epochs). dr's job — setting the global winding scale — is done by
    the first bake; from then on it is a constant of the canonical frame.
    Idempotent; also drops its now-useless optimiser moments.
    """
    model.dr_per_winding_logit.requires_grad_(False)
    if optimiser is not None:
        optimiser.state.pop(model.dr_per_winding_logit, None)


def lr_warmup_factor(iteration, warmup_start, warmup_steps):
    """Post-reset learning-rate warm-up multiplier for ``iteration``.

    Each reset zeroes the live parameters but inherits the tail of the
    global decay schedule; a short linear ramp (from 1/warmup_steps up to 1
    over warmup_steps optimisation steps after ``warmup_start``) lets the
    fresh epoch take small steps while its Adam moments re-estimate, instead
    of jumping at full late-schedule LR from a cold start. Returns 1.0
    whenever no warm-up is active.
    """
    if warmup_start is None or warmup_steps <= 0:
        return 1.0
    steps_since = int(iteration) - int(warmup_start)
    if steps_since < 0 or steps_since >= int(warmup_steps):
        return 1.0
    return (steps_since + 1) / int(warmup_steps)


def composed_slice_to_spiral_transform(frozen_models, live_slice_to_spiral):
    """True-scroll -> canonical map through every frozen epoch, then live.

    ``frozen_models`` are ordered earliest epoch first (the one adjacent to
    true scroll space). Each stage maps its epoch's input space to its baked
    output space, so applying them in order, ending with the live transform,
    reproduces the full fitted map for export and satisfaction metrics.
    """
    parts = [model.get_slice_to_spiral_transform() for model in frozen_models]
    parts.append(live_slice_to_spiral)
    return pyro.distributions.transforms.ComposeTransform(parts)


def bake_refusal_reasons(*, interactive, influence_active, phase_mode,
                         warmup_active=False):
    """Why a bake must be refused right now (empty when it may proceed).

    The SDT phase bundle cannot be baked (nor can its lasagna dependency),
    interactive sessions ingest scroll-space inputs and masks mid-run,
    influence grad masks are derived on the pre-bake flow lattice, and a
    truncated (warm-up) transform is not the transform the constraints were
    fitted against.

    Dense lasagna supervision (normals / grad-mag) is deliberately NOT a
    refusal: its volumes describe true scroll space and cannot be baked, but
    the losses stay valid for as long as the inputs are still in that space
    — so the caller runs them until the first bake and disables them there
    (FitContext._disable_lasagna_losses_for_baked_inputs) instead of
    refusing resets outright.
    """
    reasons = []
    if interactive:
        reasons.append(
            'interactive sessions are not supported: mid-run input '
            'ingestion and previews assume scroll space')
    if influence_active:
        reasons.append(
            'an influence window is active: its gradient masks live on the '
            'pre-bake flow lattice')
    if phase_mode:
        reasons.append(
            'the SDT phase bundle is enabled: the surf-SDT store cannot be '
            'baked')
    if warmup_active:
        reasons.append(
            'transform warm-up truncation is active: constraints must be '
            'baked through the untruncated transform')
    return reasons
