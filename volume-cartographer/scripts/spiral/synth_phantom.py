"""Synthetic known-deformation phantom generator for the spiral fit.

Generates a scroll-like test volume by sampling a *random known* deformation
using the production transform stack itself (SpiralAndTransform: gap expander,
integrated-flow diffeomorphism, per-z linear, umbilicus), then rendering the
canonical Archimedean spiral through it. Because the deformation is known, the
phantom ships with exact per-voxel ground truth that no real scroll has:

  volume.tif    uint8   CT-like rendered volume [Z, Y, X]
  winding.tif   float32 true fractional winding number at every voxel
  mask.tif      uint8   0 = outside annulus, 1 = valid, 2 = valid but torn
  phantom_checkpoint.pt  fit-shaped checkpoint ('spiral_and_transform', 'cfg',
                         'z_begin', 'z_end', 'spiral_outward_sense') plus
                         'umbilicus_zyx' (UmbilicusTransform keeps its curve as
                         plain attributes, so it is NOT in the state_dict) and
                         a 'phantom' block recording seed + knobs
  meta.json     provenance: seed, knobs, inverse round-trip error

Ground-truth convention: truth is the *forward* slice->spiral map exactly as
integrated (the same RK4 the fit uses), so winding.tif is exact by definition;
only users of transform.inv see numerical inversion error, which is measured
and reported in meta.json ('roundtrip_*').

Honest realism caveats, staged deliberately:
  - Tears are signal dropout only (intensity set to background inside a random
    box); the geometry stays a smooth diffeomorphism. Topological tears --
    where the winding field itself is cut -- are NOT modelled, matching the
    (known) limitation of the production transform.
  - Haze is a uniform Gaussian blur, not a physically-motivated decoherence
    model; compression-dependent haze is future work.
A method that aces this phantom can still fail on real scrolls; the phantom
bounds correctness from below, it does not certify it.
"""

import json
import os

import click
import numpy as np
import scipy.ndimage
import tifffile
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sample_spiral
from config import Config
from sample_spiral import get_theta_and_radii
from transforms import SpiralAndTransform

PHANTOM_SCHEMA_VERSION = 1


def make_phantom_config(z_size, yx_size, dr_per_winding, num_table_windings, z_margin):
    """A full fit Config with the model_* keys sized for a small phantom.

    Starting from the real Config (rather than a hand-rolled dict) keeps the
    phantom checkpoint loadable by any tool that reads fit checkpoints.
    """
    cfg = Config().as_dict()
    cfg['model_initial_dr_per_winding'] = float(dr_per_winding)
    cfg['model_flow_bounds_radius'] = yx_size // 2
    cfg['model_flow_bounds_z_margin'] = z_margin
    # Keep every parameter lattice non-degenerate at phantom sizes: the fit's
    # default resolutions assume full-scroll z extents and integer-divide to
    # zero-size lattices on small volumes (CartesianFlowField's LR lattice is
    # resolution // 6 per axis; the linear and gap lattices are z extent //
    # their respective resolutions).
    z_extent = z_size + 2 * z_margin
    cfg['model_flow_voxel_resolution'] = max(1, min(
        z_extent // 6, yx_size // 6, cfg['model_flow_voxel_resolution']))
    cfg['model_linear_z_resolution'] = min(z_extent, cfg['model_linear_z_resolution'])
    cfg['model_gap_expander_logit_resolution'] = min(
        z_extent, cfg['model_gap_expander_logit_resolution'])
    cfg['model_gap_expander_num_windings'] = num_table_windings
    return cfg


def build_model(cfg, z_begin, z_end, umbilicus_zyx, spiral_outward_sense, device):
    """Construct SpiralAndTransform the same way find_inconsistent_windings
    rebuilds one from a checkpoint (flow box from cfg bounds + model z-range)."""
    r = cfg['model_flow_bounds_radius']
    flow_min = torch.tensor([z_begin - cfg['model_flow_bounds_z_margin'], -r, -r],
                            dtype=torch.int64, device=device)
    flow_max = torch.tensor([z_end + cfg['model_flow_bounds_z_margin'], r, r],
                            dtype=torch.int64, device=device)
    model = SpiralAndTransform(
        flow_integration_steps=cfg['model_num_flow_integration_steps'],
        flow_integration_solver=cfg['model_flow_integration_solver'],
        flow_min_corner_zyx=flow_min,
        flow_max_corner_zyx=flow_max,
        umbilicus_zyx=umbilicus_zyx.to(device),
        config=cfg,
        spiral_outward_sense=spiral_outward_sense,
    )
    model.to(device)
    model.eval()
    return model


def build_model_from_phantom_checkpoint(checkpoint, device):
    """Rebuild the exact phantom transform from a phantom_checkpoint.pt payload."""
    model = build_model(
        checkpoint['cfg'],
        int(checkpoint['z_begin']),
        int(checkpoint['z_end']),
        checkpoint['umbilicus_zyx'].to(device),
        checkpoint['spiral_outward_sense'],
        device,
    )
    model.load_state_dict(checkpoint['spiral_and_transform'])
    return model


def _smooth_noise_like(param, rng, sigma):
    """Zero-mean unit-std noise of param's shape, Gaussian-smoothed over the
    trailing spatial axes so the sampled fields are smooth on the lattice."""
    noise = rng.standard_normal(tuple(param.shape)).astype(np.float32)
    spatial_axes = tuple(range(param.ndim - len(param.shape[2:]), param.ndim)) if param.ndim > 2 else tuple(range(param.ndim))
    if sigma > 0:
        noise = scipy.ndimage.gaussian_filter(noise, sigma=sigma, axes=spatial_axes)
    std = noise.std()
    if std > 0:
        noise /= std
    return torch.from_numpy(noise)


@torch.no_grad()
def randomise_model(model, rng, flow_std, gap_log_std, linear_std, flow_smooth_sigma):
    """Fill the model's parameters with a random smooth deformation.

    Magnitude knobs are expressed in each stage's *effective* units and divided
    by the training-time learning-rate scales, so knob values survive changes
    to those scales:
      flow_std       velocity in normalised flow-box units (~= displacement as
                     a fraction of the flow-box extent, since t spans 1)
      gap_log_std    std of the log winding-gap multiplier (0.15 -> gaps vary
                     roughly x0.74..x1.35): compression/expansion of spacing
      linear_std     std of the per-z 2x2 log-matrix L entries (M = expm(L)):
                     mild anisotropy/shear
    """
    # Randomise the HR flow lattice only (the LR one would upsample to a
    # near-constant field at phantom sizes); smoothing keeps it diffeomorphic
    # in practice at these magnitudes.
    hr_flow = model.flow_field.flows[1]
    hr_flow.copy_(_smooth_noise_like(hr_flow, rng, flow_smooth_sigma) * flow_std)
    model.flow_field.flows[0].zero_()
    for extra in model.extra_flow_fields:
        extra.flows[0].zero_()
        extra.flows[1].zero_()

    # Effective log gap = logits * lr_scale * 2e2 (GapExpandingTransform);
    # invert that so gap_log_std is in log-gap units. Logit 0 is pinned by the
    # transform itself, no need to zero it here.
    gap_logits = model.gap_expander_params.logits
    gap_scale = model.cfg['model_gap_expander_lr_scale'] * 2.e2
    gap_logits.copy_(_smooth_noise_like(gap_logits, rng, flow_smooth_sigma) * (gap_log_std / gap_scale))

    # Effective L = logits * linear_logits_scale (SpiralAndTransform).
    linear = model.linear_logits
    linear.copy_(_smooth_noise_like(linear, rng, 0.) * (linear_std / model.linear_logits_scale))

    model.requires_grad_(False)


def make_umbilicus(z_begin, z_end, z_margin, centre_yx, amplitude, rng):
    """A gently wandering central curve around the volume centre, dense in z so
    UmbilicusTransform's own sigma=75 smoothing has samples to work with."""
    zs = np.arange(z_begin - z_margin, z_end + z_margin, dtype=np.float32)
    wiggle = rng.standard_normal((len(zs), 2)).astype(np.float32)
    wiggle = scipy.ndimage.gaussian_filter1d(wiggle, sigma=24., axis=0, mode='nearest')
    std = wiggle.std(axis=0, keepdims=True)
    wiggle = wiggle / np.where(std > 0, std, 1.) * amplitude
    yx = centre_yx[None, :] + wiggle
    return torch.from_numpy(np.concatenate([zs[:, None], yx], axis=-1))


@torch.no_grad()
def render(model, z_size, yx_size, first_winding, last_winding, sheet_sigma,
           device, chunk_size=200_000):
    """Render the deformed spiral into slice space, with exact per-voxel truth.

    Returns (volume [0,1] float32, winding float32, valid mask bool), each
    shaped [Z, Y, X]. Winding truth is shifted_radius / dr -- the same readout
    find_inconsistent_windings calls "the model's raw winding number".
    """
    transform = model.get_slice_to_spiral_transform()
    dr = model.get_dr_per_winding()

    zs = torch.arange(z_size, dtype=torch.float32)
    ys = torch.arange(yx_size, dtype=torch.float32)
    grid = torch.cartesian_prod(zs, ys, ys)  # [N, 3] zyx, voxel centres

    density = torch.empty(len(grid), dtype=torch.float32)
    winding = torch.empty(len(grid), dtype=torch.float32)
    for start in tqdm(range(0, len(grid), chunk_size), desc='render'):
        chunk = grid[start : start + chunk_size].to(device)
        spiral = transform(chunk)
        _, _, shifted = get_theta_and_radii(spiral[..., 1:], dr)
        winding[start : start + chunk_size] = (shifted / dr).cpu()
        density[start : start + chunk_size] = sample_spiral.get_spiral_density(
            spiral[..., 1:], dr_per_winding=dr, sigma=sheet_sigma,
            winding_range=(first_winding, last_winding + 1),
        ).cpu()

    shape = (z_size, yx_size, yx_size)
    winding = winding.view(shape).numpy()
    density = density.view(shape).numpy()
    valid = (winding >= first_winding) & (winding <= last_winding)
    return density, winding, valid


def apply_tears(volume, valid, num_tears, rng, background):
    """Signal-dropout tears: random boxes forced to background intensity.
    Geometry (and winding truth) is untouched -- see module docstring. Boxes
    are centred on random valid voxels: the annulus is a thin ring, so a
    volume-uniform box would usually miss the papyrus entirely."""
    tear_mask = np.zeros(volume.shape, dtype=bool)
    z_size, y_size, x_size = volume.shape
    valid_idx = np.stack(np.nonzero(valid), axis=-1)
    for _ in range(num_tears):
        cz, cy, cx = valid_idx[rng.integers(0, len(valid_idx))]
        dz = rng.integers(z_size // 4, z_size + 1)
        dy = rng.integers(8, max(9, y_size // 8))
        dx = rng.integers(8, max(9, x_size // 8))
        box = np.zeros_like(tear_mask)
        box[max(0, cz - dz // 2) : cz + dz // 2 + 1,
            max(0, cy - dy // 2) : cy + dy // 2 + 1,
            max(0, cx - dx // 2) : cx + dx // 2 + 1] = True
        tear_mask |= box & valid
    volume[tear_mask] = background
    return tear_mask


@torch.no_grad()
def measure_roundtrip(model, valid, num_points, rng, device):
    """Slice -> spiral -> slice numerical inversion error on random valid voxels."""
    idx = np.stack(np.nonzero(valid), axis=-1)
    sel = idx[rng.integers(0, len(idx), size=min(num_points, len(idx)))]
    pts = torch.from_numpy(sel.astype(np.float32)).to(device)
    transform = model.get_slice_to_spiral_transform()
    err = (transform.inv(transform(pts)) - pts).norm(dim=-1).cpu().numpy()
    return {
        'roundtrip_p50_vox': float(np.percentile(err, 50)),
        'roundtrip_p95_vox': float(np.percentile(err, 95)),
        'roundtrip_max_vox': float(err.max()),
    }


@click.command()
@click.option('--out', required=True, type=click.Path(file_okay=False),
              help='Output directory for the phantom files.')
@click.option('--seed', default=0, type=int, help='RNG seed; same seed -> same phantom.')
@click.option('--z-size', default=64, type=int, help='Volume z extent in voxels.')
@click.option('--yx-size', default=384, type=int, help='Volume y and x extent in voxels.')
@click.option('--dr-per-winding', default=16., type=float,
              help='Canonical winding spacing in voxels (fit default: 16).')
@click.option('--first-winding', default=2, type=int,
              help='Innermost rendered winding (leaves a hollow core, like real scrolls).')
@click.option('--flow-std', default=0.01, type=float,
              help='Diffeomorphism velocity std in normalised flow-box units.')
@click.option('--gap-log-std', default=0.15, type=float,
              help='Std of log winding-gap multiplier (spacing compression/expansion).')
@click.option('--linear-std', default=0.08, type=float,
              help='Std of the per-z linear-stage log-matrix entries.')
@click.option('--flow-smooth-sigma', default=1.5, type=float,
              help='Gaussian smoothing (lattice cells) applied to sampled fields.')
@click.option('--umbilicus-amplitude', default=3., type=float,
              help='Std of the umbilicus wander around the volume centre, in voxels.')
@click.option('--sheet-sigma', default=1.5, type=float,
              help='Rendered sheet half-thickness (Gaussian sigma, voxels).')
@click.option('--noise-std', default=0.03, type=float, help='Additive Gaussian noise std.')
@click.option('--haze-sigma', default=0., type=float,
              help='Uniform Gaussian blur sigma in voxels (0 disables).')
@click.option('--num-tears', default=0, type=int,
              help='Random signal-dropout tear boxes (geometry untouched).')
@click.option('--spiral-outward-sense', default='CW', type=click.Choice(['CW', 'ACW']))
@click.option('--device', default='cpu', help="torch device ('cpu', 'cuda', ...).")
def main(out, seed, z_size, yx_size, dr_per_winding, first_winding, flow_std,
         gap_log_std, linear_std, flow_smooth_sigma, umbilicus_amplitude,
         sheet_sigma, noise_std, haze_sigma, num_tears, spiral_outward_sense, device):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device(device)

    z_margin = max(8, z_size // 4)
    centre = np.array([yx_size / 2., yx_size / 2.], dtype=np.float32)
    # Keep the outermost rendered winding clear of the volume edge and of the
    # umbilicus wander, so the annulus is fully observed.
    max_radius = yx_size / 2. - 4. * umbilicus_amplitude - 2. * dr_per_winding
    last_winding = int(max_radius / dr_per_winding) - 1
    if last_winding <= first_winding:
        raise click.UsageError('volume too small for the requested dr-per-winding')

    cfg = make_phantom_config(z_size, yx_size, dr_per_winding,
                              num_table_windings=last_winding + 8, z_margin=z_margin)
    umbilicus_zyx = make_umbilicus(0, z_size, z_margin, centre, umbilicus_amplitude, rng)
    model = build_model(cfg, 0, z_size, umbilicus_zyx, spiral_outward_sense, device)
    randomise_model(model, rng, flow_std, gap_log_std, linear_std, flow_smooth_sigma)

    volume, winding, valid = render(
        model, z_size, yx_size, first_winding, last_winding, sheet_sigma, device)

    background = 0.15
    volume = background + volume * (0.75 - background)
    if haze_sigma > 0:
        volume = scipy.ndimage.gaussian_filter(volume, sigma=haze_sigma)
    tear_mask = apply_tears(volume, valid, num_tears, rng, background) \
        if num_tears > 0 else np.zeros_like(valid)
    if noise_std > 0:
        volume = volume + rng.standard_normal(volume.shape).astype(np.float32) * noise_std

    roundtrip = measure_roundtrip(model, valid, num_points=20_000, rng=rng, device=device)

    os.makedirs(out, exist_ok=True)
    tifffile.imwrite(os.path.join(out, 'volume.tif'),
                     (np.clip(volume, 0., 1.) * 255).astype(np.uint8))
    tifffile.imwrite(os.path.join(out, 'winding.tif'), winding.astype(np.float32))
    mask = valid.astype(np.uint8)
    mask[tear_mask] = 2
    tifffile.imwrite(os.path.join(out, 'mask.tif'), mask)

    knobs = {
        'seed': seed, 'z_size': z_size, 'yx_size': yx_size,
        'dr_per_winding': dr_per_winding, 'first_winding': first_winding,
        'last_winding': last_winding, 'flow_std': flow_std,
        'gap_log_std': gap_log_std, 'linear_std': linear_std,
        'flow_smooth_sigma': flow_smooth_sigma,
        'umbilicus_amplitude': umbilicus_amplitude, 'sheet_sigma': sheet_sigma,
        'noise_std': noise_std, 'haze_sigma': haze_sigma, 'num_tears': num_tears,
    }
    torch.save({
        'schema_version': 2,
        'phantom_schema_version': PHANTOM_SCHEMA_VERSION,
        'spiral_and_transform': model.state_dict(),
        'cfg': cfg,
        'z_begin': 0,
        'z_end': z_size,
        'spiral_outward_sense': spiral_outward_sense,
        'umbilicus_zyx': umbilicus_zyx.cpu(),
        'phantom': knobs,
    }, os.path.join(out, 'phantom_checkpoint.pt'))
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump({**knobs, **roundtrip,
                   'valid_voxels': int(valid.sum()),
                   'torn_voxels': int(tear_mask.sum())}, f, indent=2)

    click.echo(f'phantom written to {out}: windings {first_winding}..{last_winding}, '
               f"{int(valid.sum())} valid voxels, "
               f"inverse round-trip p95 {roundtrip['roundtrip_p95_vox']:.4f} vox")


if __name__ == '__main__':
    main()
