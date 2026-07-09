"""Standalone debug tool: explain why a fitted spiral checkpoint is locally *bad*
-- far from a specific patch, or high-distortion in a region -- by ranking which
input constraints are first-order responsible ("blocking" the fix).

Method (finite-difference relaxation probe, no Hessian):

 1. Define a differentiable local quality measure Q at the converged parameters
    theta0 -- either the mean scroll-space distance of one patch (optionally
    just the part inside a scroll-space ball, --q-center/--q-radius) from its
    snapped target winding, or the symmetric Dirichlet energy over such a ball
    (points sampled in the ball are quantised onto their nearest winding
    surfaces at theta0 and then held fixed).
 2. Compute g = dQ/dtheta (one backward through the transform, including the
    flow field's accumulated-grad path), and from it a descent direction d
    (optionally Adam-preconditioned using the checkpoint's optimizer second
    moments, so the probe moves in the geometry the optimizer actually used).
 3. Take small virtual steps theta± = theta0 -/+ eps*d ("improve Q" / "worsen
    Q") and evaluate every constraint's training loss at theta0 and theta± --
    per verified patch, per cross-patch pcl (relative & absolute winding), per
    unattached pcl strip, plus the family-level regularisers (sym-Dirichlet,
    dense normals/spacing, umbilicus, shell).
 4. Rank constraints by the central slope (loss(improve) - loss(worsen)) / 2:
    positive = the constraint's loss rises when the bad region is allowed to
    relax = it is actively holding the fit there (a "blocker"). At a stationary
    point the weighted slopes of blockers and pullers approximately cancel, so
    each blocker's share of the total blocking force is reported too.

Per-constraint losses reuse fit_spiral's own loss functions, made deterministic
by reseeding numpy/torch with a per-(constraint, replicate) seed before each
call: the exact same sample points are drawn at theta0 and theta±, so each
replicate's loss delta is noise-free and uses the same code path as training
(straight-strip sampling; the 'dijkstra' strip pools are not replicated here).
--replicates R redraws each constraint's remaining stochastic freedoms (strip
choice within a patch, the point chosen per patch side of a winding pair, the
random L-routes per anchor) R times, averaging the slopes and reporting their
spread as a per-row noise bar. Caveats: the per-object effective weights
for the DT / winding families are first-order approximations of each object's
share of the family loss (those families aggregate nonlinearly across objects);
DBM tracks and unverified patches are not probed (disabled in current configs).

If the top blocker is a regulariser (e.g. sym_dirichlet), it is likely a
middleman transmitting force from constraints pinning the *neighbouring*
windings: re-run with --q-measure sym-dirichlet, centring the ball on that
neighbourhood, to follow the chain one hop.

Run from the spiral/ directory, e.g.

    python constraint_blame.py \
        --checkpoint out/.../checkpoint_fitted.ckpt \
        --patches-dir /path/to/dataset/patches \
        --pcl /path/to/rel_new.json --pcl /path/to/new_same_wind.json \
        --z-range 10500,11500 \
        --q-measure patch-dt --q-patch-id <patch-folder-name> \
        --q-center 3800,4100,11000 --q-radius 300

    python constraint_blame.py ... \
        --q-measure sym-dirichlet --q-center 3800,4100,11000 --q-radius 300
"""

import copy
import json
import os
import sys
import zlib

import click
import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
import fit_spiral as fs
from lasagna_data import prepare_lasagna_volume
from sample_spiral import get_theta_and_radii
from losses import (
    _patch_radius_and_dt_losses,
    _sample_patch_tracks,
    configure_losses,
    get_lasagna_losses,
    get_patch_abs_winding_loss,
    get_patch_and_umbilicus_losses,
    get_patch_rel_winding_loss,
    get_shell_outer_loss,
    get_symmetric_dirichlet_loss,
    get_unattached_pcl_strip_losses,
)


# ==========================================================================
# Checkpoint / dataset loading (modelled on find_inconsistent_windings.py)
# ==========================================================================

def resolve_umbilicus_path(patches_dir, explicit_path=None):
    if explicit_path:
        return os.path.abspath(os.path.expanduser(explicit_path))
    patches_dir = os.path.abspath(os.path.expanduser(patches_dir))
    inferred = os.path.join(os.path.dirname(patches_dir), 'umbilicus.json')
    if os.path.exists(inferred):
        return inferred
    default_dataset_path = getattr(fs, 'dataset_path', None)
    if default_dataset_path:
        default_path = os.path.join(os.path.expanduser(default_dataset_path), 'umbilicus.json')
        if os.path.exists(default_path):
            return os.path.abspath(default_path)
    raise SystemExit(
        f'could not find umbilicus.json next to --patches-dir ({inferred}); '
        'pass --umbilicus /path/to/umbilicus.json'
    )


def install_globals(checkpoint, patches_dir, pcl_paths, fibers_dir, shell_dir,
                    filter_z_begin, filter_z_end, umbilicus_path):
    """Point fit_spiral's module globals at the checkpoint's cfg and the
    user-supplied inputs so fs.main(load_only_patches_and_point_collections=True)
    loads/filters exactly as training would. Returns (cfg, model_z_begin,
    model_z_end); the model z-range shapes the flow field independently of the
    filtering z-range."""
    cfg = dict(fs.default_config)
    cfg.update(checkpoint['cfg'])
    fs.cfg = cfg
    fs.verified_patches_path = patches_dir
    fs.unverified_patches_path = None
    fs.fibers_path = fibers_dir
    fs.pcl_json_paths = list(pcl_paths)
    fs.z_begin = filter_z_begin
    fs.z_end = filter_z_end
    fs.umbilicus_z_to_yx = (
        lambda path=umbilicus_path: fs.json_umbilicus_z_to_yx(path, coordinate_scale=1.0)
    )
    if shell_dir and cfg['loss_weight_shell_outer'] > 0:
        fs.shell_path = shell_dir
    else:
        fs.shell_losses_enabled = lambda: False

    model_z_begin, model_z_end = filter_z_begin, filter_z_end
    if 'z_begin' in checkpoint:
        model_z_begin, model_z_end = int(checkpoint['z_begin']), int(checkpoint['z_end'])
    return cfg, model_z_begin, model_z_end


def build_model(checkpoint, cfg, model_z_begin, model_z_end, device):
    """Reconstruct SpiralAndTransform from the checkpoint. The high-res flow
    scale is not part of the state dict; a fully-fitted checkpoint ends the LR
    ramp at its 'final' value, so pin it there to reproduce the saved map."""
    all_zs = np.arange(model_z_begin, model_z_end)
    umbilicus_fn = fs.umbilicus_z_to_yx()
    umbilicus_zyx = torch.from_numpy(
        np.concatenate([all_zs[:, None], umbilicus_fn(all_zs)], axis=-1).astype(np.float32)
    ).to(device)

    r = cfg['flow_bounds_radius']
    flow_min_corner = torch.tensor(
        [model_z_begin - cfg['flow_bounds_z_margin'], -r, -r], dtype=torch.int64, device=device)
    flow_max_corner = torch.tensor(
        [model_z_end + cfg['flow_bounds_z_margin'], r, r], dtype=torch.int64, device=device)

    model = fs.SpiralAndTransform(
        flow_integration_steps=cfg['num_flow_integration_steps'],
        flow_integration_solver=cfg['flow_integration_solver'],
        umbilicus_zyx=umbilicus_zyx,
        flow_min_corner_zyx=flow_min_corner,
        flow_max_corner_zyx=flow_max_corner,
        config=cfg,
        spiral_outward_sense=fs.spiral_outward_sense,
    )
    model.to(device)
    model.load_state_dict(checkpoint['spiral_and_transform'])
    model.eval()
    model.flow_field.flow_scales[1] = cfg['flow_field_high_res_lr_scale_final']
    return model, umbilicus_fn


def load_adam_second_moments(model, checkpoint, cfg):
    """Rebuild the optimizer with fit_spiral's exact param-group layout, load its
    checkpointed state, and return {param -> bias-corrected exp_avg_sq}. Used to
    precondition the probe direction so it moves in the same per-parameter
    geometry the optimizer trained in (raw gradients conflate the very different
    scales of the flow / gap-expander / linear parameterisations)."""
    if 'optimiser' not in checkpoint:
        return None
    flow_field_params = list(model.flow_field.parameters())
    gap_expander_params = list(model.gap_expander_params.parameters())
    linear_params = [model.linear_logits]
    grouped_ids = {id(p) for p in flow_field_params + gap_expander_params + linear_params}
    other_params = [p for p in model.parameters() if id(p) not in grouped_ids]
    param_groups = [
        {'params': other_params, 'weight_decay': 0.0},
        {'params': linear_params, 'weight_decay': 0.0},
        {'params': gap_expander_params, 'weight_decay': cfg['weight_decay_gap_expander']},
        {'params': flow_field_params, 'weight_decay': cfg['weight_decay_flow_field']},
    ]
    optimiser = torch.optim.AdamW(param_groups, lr=cfg['learning_rate'], betas=(0.9, 0.999), eps=1.e-8)
    try:
        optimiser.load_state_dict(checkpoint['optimiser'])
    except Exception as e:
        print(f'WARNING: could not load optimizer state ({e}); falling back to raw-gradient probe direction')
        return None
    beta2 = 0.999
    moments = {}
    for p in model.parameters():
        state = optimiser.state.get(p)
        if not state or 'exp_avg_sq' not in state:
            continue
        step = state.get('step', 1)
        step = float(step.item()) if torch.is_tensor(step) else float(step)
        moments[p] = state['exp_avg_sq'] / max(1. - beta2 ** step, 1.e-12)
    return moments if moments else None


# ==========================================================================
# Patch sampling caches (fit_spiral builds these inside main(); replicated
# here since the load-only path returns before cache construction)
# ==========================================================================

def _build_line_runs(line_valid):
    padded = np.concatenate([[False], line_valid, [False]]).astype(np.int8)
    diff = np.diff(padded)
    return np.where(diff == 1)[0].astype(np.int64), np.where(diff == -1)[0].astype(np.int64)


def install_sampling_cache(patch, in_roi_quad_mask_np):
    patch._sampling_valid_quad_mask_np = in_roi_quad_mask_np
    patch._sampling_valid_quad_rows = np.flatnonzero(in_roi_quad_mask_np.any(axis=1))
    patch._sampling_valid_quad_cols = np.flatnonzero(in_roi_quad_mask_np.any(axis=0))

    def _runs_per_line(mask_np, fixed_axis, valid_lines):
        los_list, his_list, cum_list = [], [], []
        for r in valid_lines:
            line = mask_np[r] if fixed_axis == 0 else mask_np[:, r]
            los, his = _build_line_runs(line)
            los_list.append(los)
            his_list.append(his)
            cum_list.append(np.cumsum(his - los))
        return los_list, his_list, cum_list

    patch._h_runs_los, patch._h_runs_his, patch._h_runs_cum = _runs_per_line(
        in_roi_quad_mask_np, 0, patch._sampling_valid_quad_rows)
    patch._v_runs_los, patch._v_runs_his, patch._v_runs_cum = _runs_per_line(
        in_roi_quad_mask_np, 1, patch._sampling_valid_quad_cols)


def quad_roi_mask(patch, z_lo, z_hi, z_margin):
    valid_quad_mask_np = patch.valid_quad_mask.cpu().numpy()
    zyxs_z_np = patch.zyxs[..., 0].cpu().numpy()
    quad_zs_np = (zyxs_z_np[:-1, :-1] + zyxs_z_np[1:, :-1] + zyxs_z_np[:-1, 1:] + zyxs_z_np[1:, 1:]) / 4
    z_in_roi = (quad_zs_np >= z_lo - z_margin) & (quad_zs_np < z_hi + z_margin)
    return valid_quad_mask_np & z_in_roi, quad_zs_np, valid_quad_mask_np


def prepare_patch_sampling_caches(patches_list, cfg, z_lo, z_hi):
    patch_areas = np.empty(len(patches_list), dtype=np.float32)
    for patch_idx, patch in enumerate(patches_list):
        mask, _, valid_quad = quad_roi_mask(patch, z_lo, z_hi, cfg['patch_loss_z_margin'])
        if not mask.any():
            mask = valid_quad
        install_sampling_cache(patch, mask)
        patch_areas[patch_idx] = float(patch.area)
    inv_weights = patch_areas ** 0.5
    return inv_weights / inv_weights.sum()


# ==========================================================================
# Deterministic re-seeding: the same seed before a loss call at theta0 and at
# theta± draws the identical sample set, so loss deltas carry no sampling noise
# ==========================================================================

def reseed(*parts):
    h = zlib.crc32('|'.join(str(p) for p in parts).encode()) & 0x7fffffff
    np.random.seed(h)
    torch.manual_seed(h)


# ==========================================================================
# Quality measures Q
# ==========================================================================

def parse_center_xyz(text):
    try:
        x, y, z = (float(v) for v in text.split(','))
    except Exception:
        raise click.BadParameter('--q-center must be "x,y,z" (full-resolution scroll voxels)')
    return np.array([z, y, x], dtype=np.float32)  # zyx internally, like the rest of the codebase


def make_patch_q(measure, patches_list, patch_idx, atlas, cfg, z_lo, z_hi,
                 center_zyx, q_radius, num_tracks, seed):
    """Q = the target patch's radius / DT loss with zero margin and p=1 norms:
    'patch-dt' is the mean scroll-space distance (voxels) of the patch samples
    from their snapped target winding surface -- 'how far is the fit from this
    patch'; 'patch-radius' is the mean spiral-space deviation of each strip from
    constant shifted-radius. Optionally restricted to a scroll-space ball: only
    quads whose centre (mean of the 4 corner vertices) lies within q_radius of
    center_zyx are sampled, via a windowed copy of the patch's sampling cache."""
    patch = patches_list[patch_idx]
    mask, _, _ = quad_roi_mask(patch, z_lo, z_hi, cfg['patch_loss_z_margin'])
    if center_zyx is not None:
        zyxs = patch.zyxs.cpu().numpy()
        quad_centres = (zyxs[:-1, :-1] + zyxs[1:, :-1] + zyxs[:-1, 1:] + zyxs[1:, 1:]) / 4
        dist = np.linalg.norm(quad_centres - center_zyx[None, None, :], axis=-1)
        mask = mask & (dist <= q_radius)
    if not mask.any():
        raise SystemExit('no valid quads of the target patch inside the Q ball; '
                         'check --q-center (x,y,z) / increase --q-radius')
    q_patch = copy.copy(patch)  # shallow copy; only the _sampling_* attrs are rebound
    install_sampling_cache(q_patch, mask)
    q_patches_list = list(patches_list)
    q_patches_list[patch_idx] = q_patch
    q_indices = np.full(num_tracks, patch_idx, dtype=np.int64)

    def q_fn(transform, dr):
        reseed(seed, 'q-sample')
        all_slice, all_spiral, all_theta, all_shifted, _ = _sample_patch_tracks(
            transform, dr, q_patches_list, atlas, q_indices)
        radius_dev, dt_loss = _patch_radius_and_dt_losses(
            transform, dr,
            all_slice, all_spiral, all_theta, all_shifted,
            num_tracks, num_tracks, measure == 'patch-dt', None,
            0.0, False, 1.0,  # radius: margin 0, spiral-space, p=1 (plain mean)
            0.0, 1.0, 1.0,    # dt: margin 0, across p=1, within p=1 (mean distance)
        )
        return dt_loss if measure == 'patch-dt' else radius_dev

    return q_fn, int(mask.sum())


def make_sym_dirichlet_q(transform0, cfg, dr0, center_zyx, q_radius, num_points,
                         outer_winding_idx, seed, device):
    """Q = symmetric Dirichlet energy localised to a scroll-space ball: sample
    points uniformly inside the ball, map them to spiral space with the theta0
    transform, and quantise each to its nearest winding surface (snap the shifted
    radius to the nearest integer winding, keeping the point's own spiral z and
    theta). The quantised spiral points and their in-surface frames are fixed
    (built once at theta0), so only the transform varies across probe states.
    Points quantised to the core (winding < 1) or beyond the outer winding are
    dropped. The energy mirrors get_symmetric_dirichlet_loss, including its
    per-sample regularisation and cap."""
    rng = np.random.default_rng(zlib.crc32(f'{seed}|q-sd-ball'.encode()))
    directions = rng.normal(size=[num_points, 3])
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    radii = q_radius * rng.random(num_points) ** (1.0 / 3.0)  # uniform in the ball
    scroll_zyx = torch.from_numpy(
        (center_zyx[None] + directions * radii[:, None]).astype(np.float32)).to(device)
    with torch.no_grad():
        spiral0 = transform0(scroll_zyx)
    theta, _, shifted = get_theta_and_radii(spiral0[..., 1:], dr0)
    winding = torch.round(shifted / dr0)
    keep = winding >= 1
    if outer_winding_idx is not None:
        keep &= winding <= float(outer_winding_idx)
    num_kept = int(keep.sum())
    if num_kept < 16:
        raise SystemExit(f'only {num_kept} ball samples quantise onto windings; '
                         'check --q-center (x,y,z) / increase --q-radius or --q-num-points')
    theta = theta[keep]
    radius_q = (winding[keep] + theta / (2 * torch.pi)) * dr0
    spiral_zyx = torch.stack(
        [spiral0[keep][:, 0], torch.sin(theta) * radius_q, torch.cos(theta) * radius_q], dim=-1)

    dr_dtheta = dr0 / (2 * torch.pi)
    tangent_y = torch.cos(theta) * radius_q + torch.sin(theta) * dr_dtheta
    tangent_x = -torch.sin(theta) * radius_q + torch.cos(theta) * dr_dtheta
    tangential_yx = torch.nn.functional.normalize(torch.stack([tangent_y, tangent_x], dim=-1), dim=-1)
    e1 = torch.nn.functional.pad(torch.zeros_like(tangential_yx), (1, 0), value=1.)
    e2 = torch.nn.functional.pad(tangential_yx, (1, 0), value=0.)
    epsilon = float(cfg['sym_dirichlet_finite_difference_epsilon'])

    def q_fn(transform, dr):
        combined = torch.cat([spiral_zyx, spiral_zyx + e1 * epsilon, spiral_zyx + e2 * epsilon], dim=0)
        scroll_zyx, scroll_1, scroll_2 = transform.inv(combined).chunk(3, dim=0)
        a = (scroll_1 - scroll_zyx) / epsilon
        b = (scroll_2 - scroll_zyx) / epsilon
        g11 = (a * a).sum(dim=-1)
        g22 = (b * b).sum(dim=-1)
        g12 = (a * b).sum(dim=-1)
        trace_g = g11 + g22
        det_g = g11 * g22 - g12 * g12
        inverse_eps = 1e-3
        inverse_term = (trace_g + 2.0 * inverse_eps) / (det_g + inverse_eps * trace_g + inverse_eps ** 2)
        energy = (trace_g + inverse_term - 4.0).clamp(min=0.0).clamp(max=1.e2)
        return energy.mean()

    return q_fn, num_kept


# ==========================================================================
# Probe direction: backward through the transform (flow-field grads only land
# via the accumulated-grad hook, so autograd.grad would silently miss them)
# ==========================================================================

def compute_q_gradient(model, q_fn):
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        transform = model.get_slice_to_spiral_transform()
        dr = model.get_dr_per_winding()
        q = q_fn(transform, dr)
        q.backward()
    apply_accumulated_field_grad = getattr(model.flow_field, 'apply_accumulated_field_grad', None)
    if apply_accumulated_field_grad is not None:
        apply_accumulated_field_grad()
    grads = {}
    for name, p in model.named_parameters():
        grads[name] = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
    model.zero_grad(set_to_none=True)
    return float(q.detach()), grads


def build_probe_direction(model, grads, adam_moments):
    direction = {}
    g_dot_d = 0.0
    moments_by_name = {}
    if adam_moments is not None:
        param_to_name = {id(p): name for name, p in model.named_parameters()}
        moments_by_name = {param_to_name[id(p)]: v for p, v in adam_moments.items() if id(p) in param_to_name}
    for name, _ in model.named_parameters():
        g = grads[name]
        v = moments_by_name.get(name)
        d = g / (v.sqrt() + 1.e-8) if v is not None else g.clone()
        direction[name] = d
        g_dot_d += float((g * d).sum())
    return direction, g_dot_d


class ParamShifter:
    """Set model parameters to theta0 + alpha * d, restoring exactly from saved
    originals (no drift from repeated in-place add/sub)."""

    def __init__(self, model, direction):
        self.model = model
        self.direction = direction
        self.originals = {name: p.detach().clone() for name, p in model.named_parameters()}

    def set_alpha(self, alpha):
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if alpha == 0.0:
                    p.copy_(self.originals[name])
                else:
                    p.copy_(self.originals[name] + alpha * self.direction[name])


# ==========================================================================
# Per-constraint dense evaluation
# ==========================================================================

def evaluate_all_constraints(model, ctx, tag, replicate):
    """Evaluate Q plus every constraint's loss at the model's current parameters.
    Returns {(family, key): float}. Every stochastic loss call is preceded by a
    per-(constraint, replicate) reseed: the sample set is identical across probe
    states (deltas are exact per replicate) but differs across replicates, so
    averaging replicates densifies the constraint's stochastic freedoms (which
    strips represent a patch, which point per patch side a winding pair uses,
    which L-routes each anchor takes) and their spread measures sampling noise."""
    out = {}
    with torch.no_grad():
        transform = model.get_slice_to_spiral_transform()
        dr = model.get_dr_per_winding()
        if replicate == 0:
            out[('Q', ctx['q_label'])] = float(ctx['q_fn'](transform, dr))

        seed = ctx['seed']
        cfg = ctx['cfg']
        compute_dt = ctx['compute_patch_dt']

        for idx, patch_id in enumerate(tqdm(ctx['patch_ids'], desc=f'{tag}: patches', leave=False)):
            onehot = np.zeros(len(ctx['patch_ids']), dtype=np.float64)
            onehot[idx] = 1.0
            reseed(seed, replicate, 'patch', idx)
            radius_loss, umbilicus_loss, dt_loss, _ = get_patch_and_umbilicus_losses(
                transform, dr,
                ctx['tracks_per_patch'], ctx['tracks_per_patch'],
                ctx['patches_list'], ctx['atlas'], onehot, ctx['umbilicus_zyx'],
                compute_dt=compute_dt,
                dt_max_winding=ctx['patch_dt_max_winding'],
            )
            out[('patch_radius', patch_id)] = float(radius_loss)
            if compute_dt:
                out[('patch_dt', patch_id)] = float(dt_loss)
            if idx == 0 and cfg['loss_weight_umbilicus'] > 0:
                out[('umbilicus', '(all)')] = float(umbilicus_loss)

        if cfg['loss_weight_rel_winding'] > 0:
            for i, (label, pcl) in enumerate(tqdm(ctx['rel_pcls'], desc=f'{tag}: rel winding', leave=False)):
                reseed(seed, replicate, 'rel', i)
                out[('rel_winding', label)] = float(get_patch_rel_winding_loss(
                    transform, dr, ctx['patches_dict'], ctx['atlas'], [pcl]))

        if cfg['loss_weight_abs_winding'] > 0:
            for i, (label, pcl) in enumerate(tqdm(ctx['abs_pcls'], desc=f'{tag}: abs winding', leave=False)):
                reseed(seed, replicate, 'abs', i)
                out[('abs_winding', label)] = float(get_patch_abs_winding_loss(
                    transform, dr, ctx['patches_dict'], ctx['atlas'], [pcl]))

        if cfg['loss_weight_unattached_pcl_radius'] > 0 or cfg['loss_weight_unattached_pcl_dt'] > 0:
            for i, (label, strip_list, n_points) in enumerate(
                    tqdm(ctx['unattached_lists'], desc=f'{tag}: unattached pcls', leave=False)):
                reseed(seed, replicate, 'unattached', i)
                radius_loss, dt_loss = get_unattached_pcl_strip_losses(
                    transform, dr, strip_list, fs.get_or_build_unattached_pcl_flat,
                    1, n_points,
                    compute_dt=compute_dt,
                    dt_max_winding=ctx['patch_dt_max_winding'],
                )
                out[('unattached_pcl_radius', label)] = float(radius_loss)
                if compute_dt:
                    out[('unattached_pcl_dt', label)] = float(dt_loss)

        if cfg['loss_weight_sym_dirichlet'] > 0:
            reseed(seed, replicate, 'sym-dirichlet')
            out[('sym_dirichlet', '(all)')] = float(get_symmetric_dirichlet_loss(
                transform, dr, ctx['outer_winding_idx'], ctx['regularisation_num_points']))

        if ctx['lasagna_volume'] is not None:
            reseed(seed, replicate, 'lasagna')
            normals_loss, spacing_loss = get_lasagna_losses(
                transform, dr, ctx['lasagna_volume'], ctx['outer_winding_idx'],
                cfg['dense_normals_num_points'])
            if cfg['loss_weight_dense_normals'] > 0:
                out[('dense_normals', '(all)')] = float(normals_loss)
            if cfg['loss_weight_dense_spacing'] > 0:
                out[('dense_spacing', '(all)')] = float(spacing_loss)

        if ctx['shell_map'] is not None:
            reseed(seed, replicate, 'shell')
            shell_loss, _ = get_shell_outer_loss(
                ctx['shell_map'], transform, dr, ctx['outer_winding_idx'])
            out[('shell_outer', '(all)')] = float(shell_loss)

    return out


def effective_weight(family, key, ctx):
    """Approximate share of the total training-loss pressure this constraint
    object exerts: family loss weight x the object's expected sampling share.
    Exact for the linearly-averaged families; first-order for the power-mean
    (DT) and pooled-pair (winding) aggregations."""
    cfg = ctx['cfg']
    if family == 'patch_radius':
        return cfg['loss_weight_patch_radius'] * ctx['patch_prob_by_id'][key]
    if family == 'patch_dt':
        return cfg['loss_weight_patch_dt'] * ctx['patch_prob_by_id'][key]
    if family == 'rel_winding':
        return cfg['loss_weight_rel_winding'] / max(len(ctx['rel_pcls']), 1)
    if family == 'abs_winding':
        return cfg['loss_weight_abs_winding'] / max(len(ctx['abs_pcls']), 1)
    if family == 'unattached_pcl_radius':
        return cfg['loss_weight_unattached_pcl_radius'] / max(len(ctx['unattached_lists']), 1)
    if family == 'unattached_pcl_dt':
        return cfg['loss_weight_unattached_pcl_dt'] / max(len(ctx['unattached_lists']), 1)
    return {
        'umbilicus': cfg['loss_weight_umbilicus'],
        'sym_dirichlet': cfg['loss_weight_sym_dirichlet'],
        'dense_normals': cfg['loss_weight_dense_normals'],
        'dense_spacing': cfg['loss_weight_dense_spacing'],
        'shell_outer': cfg['loss_weight_shell_outer'],
        'Q': 0.0,
    }[family]


# ==========================================================================
# CLI
# ==========================================================================

def pcl_label(index, pcl):
    name = pcl.get('name') or f'pcl-{index}'
    source = os.path.basename(pcl.get('source_file') or '?')
    return f'{name} [{source}] #{index}'


@click.command()
@click.option('--checkpoint', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to a checkpoint_*.ckpt saved by fit_spiral.')
@click.option('--patches-dir', required=True, type=click.Path(exists=True, file_okay=False),
              help='Directory of verified patch tifxyz folders (the set used for training).')
@click.option('--umbilicus', default=None, type=click.Path(exists=True, dir_okay=False),
              help='Path to umbilicus.json. Defaults to umbilicus.json next to --patches-dir.')
@click.option('--pcl', 'pcl_paths', required=True, multiple=True, type=click.Path(exists=True, dir_okay=False),
              help='Point-collection json file(s), as passed to training. Repeat --pcl for several.')
@click.option('--fibers-dir', default=None, type=click.Path(file_okay=False),
              help='Directory of fiber point-collection jsons (training default: dataset/fibers). '
                   'Omit to skip fibers.')
@click.option('--shell-dir', default=None, type=click.Path(file_okay=False),
              help='Shell tifxyz directory; enables the shell_outer constraint row when the '
                   'checkpoint trained with loss_weight_shell_outer > 0. Omit to skip.')
@click.option('--lasagna/--no-lasagna', default=True,
              help='Include the dense normals/spacing constraint rows, loading the normal/grad_mag '
                   'zarrs from fit_spiral module defaults (heavy; skipped automatically if the '
                   'paths are missing or the checkpoint weights are 0).')
@click.option('--z-range', required=True,
              help='Constraint filtering z-range "zlo,zhi" (full-resolution scan z, like fit_spiral '
                   'z_begin/z_end). Must lie within the checkpoint z-range.')
@click.option('--q-measure', type=click.Choice(['patch-dt', 'patch-radius', 'sym-dirichlet']),
              default='patch-dt', show_default=True,
              help='Quality measure Q: mean scroll-space distance of a patch from its snapped target '
                   'winding (patch-dt), mean spiral-space strip non-constancy of a patch '
                   '(patch-radius), or symmetric Dirichlet energy over a spiral-space window '
                   '(sym-dirichlet).')
@click.option('--q-patch-id', default=None,
              help='Target patch folder name (required for patch-dt / patch-radius).')
@click.option('--q-center', default=None,
              help='"x,y,z" centre of the Q localisation ball, in full-resolution scroll voxels. '
                   'Optional for the patch measures (whole patch when omitted); required for '
                   'sym-dirichlet.')
@click.option('--q-radius', default=None, type=float,
              help='Radius of the Q localisation ball (scroll voxels). Given together with '
                   '--q-center.')
@click.option('--precondition', type=click.Choice(['adam', 'none']), default='adam', show_default=True,
              help='Probe direction preconditioning. adam uses the checkpoint optimizer second '
                   'moments (recommended: matches the geometry the fit was optimised in).')
@click.option('--target-q-drop', default=0.1, show_default=True, type=float,
              help='Pick the probe step size so the first-order predicted relative drop in Q is '
                   'this fraction. Ignored when --epsilon is given.')
@click.option('--epsilon', default=None, type=float,
              help='Explicit probe step size along the (preconditioned) descent direction.')
@click.option('--tracks-per-patch', default=16, show_default=True, type=int,
              help='Row/column strips sampled per patch per per-constraint evaluation.')
@click.option('--q-tracks', default=64, show_default=True, type=int,
              help='Strips sampled for the patch quality measures.')
@click.option('--q-num-points', default=20000, show_default=True, type=int,
              help='Sample points for the sym-dirichlet quality measure.')
@click.option('--points-per-strip', default=96, show_default=True, type=int,
              help='Points per unattached-pcl strip per evaluation.')
@click.option('--replicates', default=8, show_default=True, type=int,
              help='Independent sample draws per constraint per probe state. Each replicate '
                   'redraws the stochastic freedoms the losses do not enumerate (which strips '
                   'represent a patch, which point per patch side a winding pair uses, which '
                   'L-routes each anchor takes); slopes are averaged and their spread is '
                   'reported per row. Evaluation time scales linearly.')
@click.option('--outer-winding', default=None, type=int,
              help='Override shell_outer_winding_idx (needed by sym-dirichlet / lasagna families) '
                   'when the checkpoint cfg has none.')
@click.option('--seed', default=0, show_default=True, type=int, help='Base seed for all sampling.')
@click.option('--top', default=25, show_default=True, type=int,
              help='How many top blockers / pullers to print.')
@click.option('--output', default=None, type=click.Path(dir_okay=False),
              help='Output json path (default: constraint_blame_<q>.json next to the checkpoint).')
def main(checkpoint, patches_dir, umbilicus, pcl_paths, fibers_dir, shell_dir, lasagna, z_range,
         q_measure, q_patch_id, q_center, q_radius,
         precondition, target_q_drop, epsilon, tracks_per_patch, q_tracks, q_num_points,
         points_per_strip, replicates, outer_winding, seed, top, output):
    device = torch.device('cuda')

    if replicates < 1:
        raise click.BadParameter('--replicates must be >= 1')

    z_lo, z_hi = (int(v) for v in z_range.split(','))
    if z_lo >= z_hi:
        raise click.BadParameter('--z-range must be "zlo,zhi" with zlo < zhi')
    if q_measure in ('patch-dt', 'patch-radius') and not q_patch_id:
        raise click.BadParameter(f'--q-measure {q_measure} requires --q-patch-id')
    if (q_center is None) != (q_radius is None):
        raise click.BadParameter('--q-center and --q-radius must be given together')
    if q_radius is not None and q_radius <= 0:
        raise click.BadParameter('--q-radius must be > 0')
    if q_measure == 'sym-dirichlet' and q_center is None:
        raise click.BadParameter('--q-measure sym-dirichlet requires --q-center and --q-radius')
    center_zyx = parse_center_xyz(q_center) if q_center else None

    umbilicus_path = resolve_umbilicus_path(patches_dir, umbilicus)
    print(f'using umbilicus {umbilicus_path}')

    print(f'loading checkpoint {checkpoint}')
    ckpt = torch.load(checkpoint, map_location='cpu')
    cfg, model_z_begin, model_z_end = install_globals(
        ckpt, patches_dir, pcl_paths, fibers_dir, shell_dir, z_lo, z_hi, umbilicus_path)
    print(f'checkpoint (model) z-range: [{model_z_begin}, {model_z_end}); '
          f'filtering z-range: [{z_lo}, {z_hi})')
    if not (z_lo >= model_z_begin and z_hi <= model_z_end):
        raise SystemExit(
            f'filtering z-range [{z_lo}, {z_hi}) extends beyond the checkpoint z-range '
            f'[{model_z_begin}, {model_z_end}); the transform is undefined outside its domain.')

    # Loss hyperparameters must match training; only the per-step sampling counts
    # for the winding losses are raised so each per-pcl call covers it densely.
    # 64 pairs/points fully covers pcls spanning up to 65 patches; both are min()-clamped
    # to each pcl's actual pair/point count inside the losses. Note the rel loss builds a
    # per-pair all-pairs matrix, so its transient GPU cost scales with this count.
    probe_cfg = dict(cfg)
    probe_cfg['rel_winding_num_patch_pairs_per_pcl'] = max(cfg['rel_winding_num_patch_pairs_per_pcl'], 64)
    probe_cfg['abs_winding_num_points_per_pcl'] = max(cfg['abs_winding_num_points_per_pcl'], 64)
    configure_losses(probe_cfg, z_lo, z_hi)

    print('loading + z-filtering patches and point-collections')
    patches, _unverified, shell_patch, cross_patch_pcls, unattached_pcl_strips = fs.main(
        load_only_patches_and_point_collections=True)
    patches_list = list(patches.values())
    patch_ids = list(patches.keys())
    if q_patch_id is not None and q_patch_id not in patches:
        raise SystemExit(f'--q-patch-id {q_patch_id!r} not among {len(patches)} loaded patches')

    patch_probabilities = prepare_patch_sampling_caches(patches_list, cfg, z_lo, z_hi)
    atlas = fs.PatchGpuAtlas(patches, device='cuda')
    print(f'{len(patches)} patches ({atlas.memory_mb():.1f} MB atlas), '
          f'{len(cross_patch_pcls)} cross-patch pcls, {len(unattached_pcl_strips)} unattached strips')

    print('rebuilding spiral transform from checkpoint')
    model, umbilicus_fn = build_model(ckpt, cfg, model_z_begin, model_z_end, device)
    adam_moments = load_adam_second_moments(model, ckpt, cfg) if precondition == 'adam' else None
    if precondition == 'adam' and adam_moments is None:
        print('WARNING: no usable optimizer state in checkpoint; using raw-gradient direction')

    # Final-equilibrium DT gating: were the DT losses on at the end of training,
    # and what was the final progressive winding cutoff (usually None / fully open)?
    num_steps = int(cfg['num_training_steps'])
    compute_patch_dt = num_steps > int(cfg['loss_start_patch_dt']) and (
        cfg['loss_weight_patch_dt'] > 0 or cfg['loss_weight_unattached_pcl_dt'] > 0)
    outer_winding_idx = outer_winding if outer_winding is not None else cfg['shell_outer_winding_idx']
    dt_gate_outer = outer_winding_idx
    patch_dt_max_winding = fs.get_progressive_dt_max_winding(
        num_steps, int(cfg['loss_start_patch_dt']), dt_gate_outer)
    if (cfg['loss_weight_sym_dirichlet'] > 0 or lasagna) and outer_winding_idx is None:
        raise SystemExit('checkpoint cfg has shell_outer_winding_idx=None; pass --outer-winding')

    # Umbilicus samples over the filter range (subsampled: every call packs them
    # into its transform batch, and the loss is a plain mean over z).
    umb_zs = np.arange(z_lo, z_hi)
    if len(umb_zs) > 256:
        umb_zs = umb_zs[np.linspace(0, len(umb_zs) - 1, 256).astype(np.int64)]
    umbilicus_zyx = torch.from_numpy(np.concatenate(
        [umb_zs[:, None], umbilicus_fn(umb_zs)], axis=-1).astype(np.float32)).to(device)

    # Constraint enumerations. rel-winding candidates mirror training's pool
    # (every multi-point cross-patch pcl, absolute ones included) but skip pcls
    # that cannot form a cross-patch pair.
    rel_pcls = [
        (pcl_label(i, pcl), pcl) for i, pcl in enumerate(cross_patch_pcls)
        if len(pcl['points']) > 1 and len(pcl.get('points_by_patch', {})) >= 2
    ]
    abs_pcls = [
        (pcl_label(i, pcl), pcl) for i, pcl in enumerate(cross_patch_pcls)
        if pcl.get('metadata', {}).get('winding_is_absolute', False) and pcl.get('points_by_patch')
    ]
    unattached_lists = []
    for i, strip in enumerate(unattached_pcl_strips):
        label = f"{strip.get('name') or strip['id']} [{os.path.basename(strip.get('source_file') or '?')}] #{strip['id']}"
        single = fs._UnattachedPclStripList([strip])
        fs.get_or_build_unattached_pcl_flat(single, device)  # build .flat once; reused across evals
        n_points = min(max(points_per_strip, 2), max(len(strip['zyxs']), 2))
        unattached_lists.append((label, single, n_points))

    lasagna_volume = None
    if lasagna and (cfg['loss_weight_dense_normals'] > 0 or cfg['loss_weight_dense_spacing'] > 0):
        try:
            lasagna_volume = prepare_lasagna_volume(
                None,
                use_normals=cfg['loss_weight_dense_normals'] > 0,
                use_spacing=cfg['loss_weight_dense_spacing'] > 0,
                normal_nx_zarr_path=fs.normal_nx_zarr_path,
                normal_ny_zarr_path=fs.normal_ny_zarr_path,
                grad_mag_zarr_path=fs.grad_mag_zarr_path,
                normal_zarr_group=fs.normal_zarr_group,
                z_begin=z_lo,
                z_end=z_hi,
                lasagna_scale=fs.lasagna_scale,
            )
        except Exception as e:
            print(f'WARNING: skipping dense normals/spacing constraints (lasagna load failed: {e})')

    shell_map = None
    if shell_patch is not None and cfg['loss_weight_shell_outer'] > 0:
        shell_map = fs.ShellPolarMap(
            shell_patch, umbilicus_fn,
            z_min=z_lo - cfg['flow_bounds_z_margin'],
            z_max=z_hi + cfg['flow_bounds_z_margin'],
            num_theta_bins=cfg['shell_num_theta_bins'],
            device=device,
        )
        if cfg['loss_weight_shell_patch_radius'] > 0:
            print('WARNING: loss_weight_shell_patch_radius > 0 is not probed by this tool')

    # --- quality measure ---
    with torch.no_grad():
        dr0 = model.get_dr_per_winding().detach().clone()

    ball_label = f'ball xyz [{q_center}] r {q_radius:g}' if q_center else None
    if q_measure in ('patch-dt', 'patch-radius'):
        q_fn, n_window_quads = make_patch_q(
            q_measure, patches_list, patch_ids.index(q_patch_id), atlas, cfg,
            z_lo, z_hi, center_zyx, q_radius, q_tracks, seed)
        q_label = f'{q_measure}({q_patch_id}' + (f', {ball_label}' if ball_label else '') + ')'
        print(f'Q = {q_label}: {n_window_quads} valid quads in ball, {q_tracks} strips per eval')
    else:
        with torch.no_grad():
            transform0 = model.get_slice_to_spiral_transform()
        q_fn, n_kept = make_sym_dirichlet_q(
            transform0, cfg, dr0, center_zyx, q_radius, q_num_points,
            outer_winding_idx, seed, device)
        q_label = f'sym-dirichlet({ball_label})'
        print(f'Q = {q_label}: {n_kept}/{q_num_points} ball samples quantised onto windings')

    # --- gradient, direction, step size ---
    print('computing dQ/dtheta')
    q0, grads = compute_q_gradient(model, q_fn)
    direction, g_dot_d = build_probe_direction(model, grads, adam_moments)
    grad_norm = float(sum((g ** 2).sum() for g in grads.values()) ** 0.5)
    print(f'Q(theta0) = {q0:.6g}; |dQ/dtheta| = {grad_norm:.4g}; <g, d> = {g_dot_d:.4g}')
    if not np.isfinite(g_dot_d) or g_dot_d <= 0:
        raise SystemExit('degenerate probe: Q has no usable gradient (check the window / measure)')
    if epsilon is None:
        epsilon = target_q_drop * max(q0, 1e-12) / g_dot_d
    predicted_drop = epsilon * g_dot_d
    print(f'probe step epsilon = {epsilon:.6g} (first-order predicted dQ = -{predicted_drop:.6g}, '
          f'{predicted_drop / max(q0, 1e-12) * 100:.1f}% of Q)')

    ctx = {
        'seed': seed,
        'cfg': probe_cfg,
        'q_fn': q_fn,
        'q_label': q_label,
        'patches_list': patches_list,
        'patches_dict': patches,
        'patch_ids': patch_ids,
        'patch_prob_by_id': dict(zip(patch_ids, patch_probabilities.astype(float))),
        'atlas': atlas,
        'umbilicus_zyx': umbilicus_zyx,
        'tracks_per_patch': tracks_per_patch,
        'compute_patch_dt': compute_patch_dt,
        'patch_dt_max_winding': patch_dt_max_winding,
        'rel_pcls': rel_pcls,
        'abs_pcls': abs_pcls,
        'unattached_lists': unattached_lists,
        'outer_winding_idx': outer_winding_idx,
        'regularisation_num_points': int(cfg['regularisation_num_points']),
        'lasagna_volume': lasagna_volume,
        'shell_map': shell_map,
    }

    # --- the three evaluation states x R replicates ---
    def run_state(tag):
        return [
            evaluate_all_constraints(model, ctx, f'{tag} [{r + 1}/{replicates}]', r)
            for r in range(replicates)
        ]

    shifter = ParamShifter(model, direction)
    print(f'evaluating constraints at theta0 ({replicates} replicate(s))')
    losses_0 = run_state('theta0')
    print('evaluating constraints at theta_improve (Q relaxed)')
    shifter.set_alpha(-epsilon)
    losses_improve = run_state('improve')
    print('evaluating constraints at theta_worsen')
    shifter.set_alpha(+epsilon)
    losses_worsen = run_state('worsen')
    shifter.set_alpha(0.0)

    q_improve = losses_improve[0][('Q', q_label)]
    q_worsen = losses_worsen[0][('Q', q_label)]
    print(f'\nQ: {q0:.6g} -> {q_improve:.6g} at theta_improve (delta {q_improve - q0:+.6g}; '
          f'first-order predicted {-predicted_drop:+.6g}), {q_worsen:.6g} at theta_worsen')
    asym = (q_improve + q_worsen - 2 * q0)
    if abs(asym) > 0.5 * abs(q_improve - q_worsen) / 2:
        print('WARNING: strong curvature in Q along the probe (second-order term is a large '
              'fraction of the slope); consider a smaller --target-q-drop')

    # --- scores ---
    # Per constraint: R replicate values per state. Each replicate's slope is an
    # exact delta for its own sample set; the replicate mean densifies the
    # constraint's sampling freedoms and the spread is a per-row noise bar.
    rows = []
    for key in losses_0[0]:
        family, name = key
        if family == 'Q':
            continue
        l0s = np.array([rep[key] for rep in losses_0])
        lps = np.array([rep.get(key, l0s[i]) for i, rep in enumerate(losses_improve)])
        lms = np.array([rep.get(key, l0s[i]) for i, rep in enumerate(losses_worsen)])
        slopes = (lps - lms) / 2.0  # >0: loss rises when Q relaxes => blocker
        slope = float(slopes.mean())
        slope_sd = float(slopes.std(ddof=1)) if replicates > 1 else 0.0
        l0, lp, lm = float(l0s.mean()), float(lps.mean()), float(lms.mean())
        w = effective_weight(family, name, ctx)
        rows.append({
            'family': family,
            'constraint': name,
            'eff_weight': w,
            'loss_0': l0,
            'delta_improve': lp - l0,
            'slope_central': slope,
            'slope_sd': slope_sd,
            'weighted_slope': w * slope,
            'weighted_slope_sd': w * slope_sd,
            'slopes': [float(s) for s in slopes],
            'curvature': lp + lm - 2 * l0,
            # Mean loss moved the same way on both sides of theta0: the response is
            # not locally linear (e.g. a DT snap target flipped winding across the
            # probe), so treat this row's slope with suspicion.
            'nonmonotone': (lp - l0) * (l0 - lm) < 0,
            # Across-replicate spread exceeds the mean slope: the score is dominated
            # by which strips / points / L-routes were drawn, not by a consistent
            # pull -- raise --replicates or treat as inconclusive.
            'noisy': replicates > 1 and slope_sd > abs(slope),
        })

    total_blocking = sum(r['weighted_slope'] for r in rows if r['weighted_slope'] > 0)
    for r in rows:
        r['blocking_share'] = r['weighted_slope'] / total_blocking if (
            total_blocking > 0 and r['weighted_slope'] > 0) else 0.0
    rows.sort(key=lambda r: r['weighted_slope'], reverse=True)

    net = sum(r['weighted_slope'] for r in rows)
    print(f'net weighted slope over all constraints: {net:+.4g} '
          f'(≈0 when the checkpoint is converged; total blocking force {total_blocking:.4g})')

    def print_rows(selected, title):
        print(f'\n=== {title} ===')
        print(f'{"family":<22} {"constraint":<52} {"w_eff":>9} {"loss@0":>11} '
              f'{"d(improve)":>11} {"w*slope":>11} {"+-sd":>10} {"share":>7}')
        for r in selected:
            flag = (' !' if r['nonmonotone'] else '') + (' ~' if r['noisy'] else '')
            print(f'{r["family"]:<22} {r["constraint"][:52]:<52} {r["eff_weight"]:>9.3g} '
                  f'{r["loss_0"]:>11.5g} {r["delta_improve"]:>+11.4g} '
                  f'{r["weighted_slope"]:>+11.4g} {r["weighted_slope_sd"]:>10.3g} '
                  f'{r["blocking_share"]:>6.1%}{flag}')
        if any(r['nonmonotone'] for r in selected):
            print('    ! = non-monotone along the probe (loss moved the same way at theta+ and '
                  'theta-, e.g. a DT snap flip); slope unreliable')
        if any(r['noisy'] for r in selected):
            print('    ~ = across-replicate spread exceeds the slope; score dominated by sampling '
                  'noise -- raise --replicates or treat as inconclusive')

    blockers = [r for r in rows if r['weighted_slope'] > 0][:top]
    pullers = [r for r in reversed(rows) if r['weighted_slope'] < 0][:top]
    print_rows(blockers, f'top blockers: constraints whose loss RISES when Q is relaxed '
                         f'(these hold the fit in its bad local state)')
    print_rows(pullers, 'top pullers: constraints that would also improve (aligned with Q)')

    family_sums = {}
    for r in rows:
        family_sums[r['family']] = family_sums.get(r['family'], 0.0) + r['weighted_slope']
    print('\nper-family weighted slope (blocking > 0):')
    for family, s in sorted(family_sums.items(), key=lambda kv: kv[1], reverse=True):
        print(f'  {family:<24} {s:+.4g}')

    if output is None:
        output = os.path.join(os.path.dirname(os.path.abspath(checkpoint)),
                              f'constraint_blame_{q_measure}.json')
    with open(output, 'w') as f:
        json.dump({
            'checkpoint': os.path.abspath(checkpoint),
            'q': {
                'measure': q_measure, 'label': q_label, 'patch_id': q_patch_id,
                'center_xyz': q_center, 'radius': q_radius,
                'value_0': q0, 'value_improve': q_improve, 'value_worsen': q_worsen,
            },
            'probe': {
                'precondition': precondition if adam_moments is not None else 'none',
                'epsilon': epsilon, 'g_dot_d': g_dot_d, 'grad_norm': grad_norm,
                'predicted_q_drop': predicted_drop, 'seed': seed,
                'replicates': replicates,
                'compute_patch_dt': compute_patch_dt,
                'patch_dt_max_winding': patch_dt_max_winding,
            },
            'net_weighted_slope': net,
            'total_blocking_force': total_blocking,
            'family_weighted_slopes': family_sums,
            'constraints': rows,
        }, f, indent=1)
    print(f'\nwrote {output}')


if __name__ == '__main__':
    main()
