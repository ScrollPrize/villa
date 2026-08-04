"""Ground-truth winding-error metrics against a synth_phantom phantom.

The spiral fit is currently scored by self-consistency (satisfaction_metrics:
agreement with its own input constraints) and by an indirect ink-coverage
proxy (get_ink_metrics). Neither compares against truth. This tool scores a
candidate winding solution against a phantom's exact per-voxel winding field:

  MAE / RMSE      of the recovered fractional winding (gauge-aligned, below)
  switch rate     fraction of points assigned to the wrong winding
                  (|aligned residual| > 0.5) -- the sheet-switch failure mode
  per-winding     MAE binned by true winding index (where does it drift?)
  torn vs clean   breakdown over the phantom's signal-dropout tear regions

Candidates:
  --candidate-checkpoint      a phantom checkpoint (self-tests, method dev) or
                              a real fit checkpoint (needs --umbilicus, loaded
                              via fit_spiral's own umbilicus reader)
  --candidate-winding-volume  a float32 TIFF of per-voxel winding on the
                              phantom's voxel grid -- so ANY method that
                              produces a winding field (e.g. a lasagna winding
                              volume) can be scored, not just spiral fits

Gauge: a candidate fit of the same volume can differ from the phantom by a
constant winding offset (theta-origin / winding-index choice), so residuals
are aligned by their median before scoring; raw (unaligned) MAE is also
reported. Opposite-sense (CW vs ACW) candidates are not auto-detected.

Branch cut: a winding field w = (r - theta*dr/2pi)/dr necessarily jumps by 1
across its theta=0 ray. When truth and candidate place that ray differently, a
thin angular band shows spurious +-1 residuals even for a perfect candidate.
Points within --cut-margin radians of either transform's cut are excluded
(excluded fraction is reported); volume candidates only get the truth-side
exclusion, since their cut placement is unknown.
"""

import json
import os

import click
import numpy as np
import tifffile
import torch

from checkpoint_io import load_checkpoint_cpu
from sample_spiral import get_theta_and_radii
from synth_phantom import build_model_from_phantom_checkpoint, build_model


def load_phantom(phantom_dir, device):
    checkpoint = load_checkpoint_cpu(os.path.join(phantom_dir, 'phantom_checkpoint.pt'))
    if 'phantom_schema_version' not in checkpoint:
        raise click.UsageError(f'{phantom_dir} is not a synth_phantom output')
    model = build_model_from_phantom_checkpoint(checkpoint, device)
    mask = tifffile.imread(os.path.join(phantom_dir, 'mask.tif'))
    return checkpoint, model, mask


def load_candidate_model(path, umbilicus_path, device):
    checkpoint = load_checkpoint_cpu(path)
    if 'umbilicus_zyx' in checkpoint:  # phantom-style checkpoint, self-contained
        return build_model_from_phantom_checkpoint(checkpoint, device)
    if umbilicus_path is None:
        raise click.UsageError(
            'real fit checkpoints do not store their umbilicus; pass --umbilicus')
    # Reuse fit_spiral's own umbilicus reader rather than re-implementing its
    # json format here; imported lazily since fit_spiral is a heavy module.
    import fit_spiral as fs
    z_begin, z_end = int(checkpoint['z_begin']), int(checkpoint['z_end'])
    all_zs = np.arange(z_begin, z_end)
    yx = fs.json_umbilicus_z_to_yx(umbilicus_path, coordinate_scale=1.0)(all_zs)
    umbilicus_zyx = torch.from_numpy(
        np.concatenate([all_zs[:, None], yx], axis=-1).astype(np.float32))
    cfg = fs.Config().as_dict()
    cfg.update(checkpoint['cfg'])
    model = build_model(cfg, z_begin, z_end, umbilicus_zyx,
                        checkpoint['spiral_outward_sense'], device)
    model.load_state_dict(checkpoint['spiral_and_transform'])
    return model


@torch.no_grad()
def winding_and_theta(model, zyx):
    """Fractional winding (shifted_radius / dr, the fit's raw winding number)
    and canonical theta for slice-space points."""
    transform = model.get_slice_to_spiral_transform()
    dr = model.get_dr_per_winding()
    spiral = transform(zyx)
    theta, _, shifted = get_theta_and_radii(spiral[..., 1:], dr)
    return (shifted / dr).cpu().numpy(), theta.cpu().numpy()


def sample_eval_points(mask, num_points, rng):
    """Random valid voxel centres; returns (zyx float32 [N,3], torn bool [N])."""
    idx = np.stack(np.nonzero(mask > 0), axis=-1)
    sel = idx[rng.integers(0, len(idx), size=min(num_points, len(idx)))]
    torn = mask[sel[:, 0], sel[:, 1], sel[:, 2]] == 2
    return sel.astype(np.float32), torn


def cut_distance(theta):
    return np.minimum(theta, 2 * np.pi - theta)


def score(w_true, w_cand, torn, cut_ok):
    """Gauge-aligned residual statistics; see module docstring for definitions."""
    residual = (w_cand - w_true)[cut_ok]
    torn = torn[cut_ok]
    offset = float(np.median(residual))
    aligned = residual - offset

    def stats(sel):
        r = aligned[sel]
        if len(r) == 0:
            return None
        return {
            'n': int(len(r)),
            'mae': float(np.abs(r).mean()),
            'rmse': float(np.sqrt((r ** 2).mean())),
            'switch_rate': float((np.abs(r) > 0.5).mean()),
        }

    per_winding = {}
    bins = np.floor(w_true[cut_ok]).astype(np.int64)
    for b in np.unique(bins):
        per_winding[int(b)] = stats(bins == b)

    return {
        'gauge_offset': offset,
        'raw_mae': float(np.abs(residual).mean()),
        'cut_excluded_frac': float(1. - cut_ok.mean()),
        'overall': stats(np.ones(len(aligned), dtype=bool)),
        'clean': stats(~torn),
        'torn': stats(torn),
        'per_winding': per_winding,
    }


def print_report(result):
    o = result['overall']
    click.echo(f"gauge offset {result['gauge_offset']:+.3f} windings; "
               f"cut-excluded {result['cut_excluded_frac'] * 100:.1f}% of samples")
    click.echo(f"overall  n={o['n']}  MAE {o['mae']:.4f}  RMSE {o['rmse']:.4f}  "
               f"switch {o['switch_rate'] * 100:.2f}%   (raw MAE {result['raw_mae']:.4f})")
    for name in ('clean', 'torn'):
        s = result[name]
        if s is not None:
            click.echo(f"{name:8s} n={s['n']}  MAE {s['mae']:.4f}  "
                       f"switch {s['switch_rate'] * 100:.2f}%")
    click.echo('per-winding MAE / switch%:')
    for b, s in sorted(result['per_winding'].items()):
        click.echo(f'  w={b:3d}  {s["mae"]:.4f}  {s["switch_rate"] * 100:6.2f}%  (n={s["n"]})')


def evaluate(phantom_model, mask, candidate, num_points, cut_margin, seed, device,
             chunk_size=200_000):
    """candidate: ('model', SpiralAndTransform) or ('volume', float32 ndarray)."""
    rng = np.random.default_rng(seed)
    zyx_np, torn = sample_eval_points(mask, num_points, rng)

    w_true = np.empty(len(zyx_np), dtype=np.float32)
    theta_true = np.empty(len(zyx_np), dtype=np.float32)
    kind, payload = candidate
    w_cand = np.empty(len(zyx_np), dtype=np.float32)
    theta_cand = None if kind == 'volume' else np.empty(len(zyx_np), dtype=np.float32)

    for start in range(0, len(zyx_np), chunk_size):
        sl = slice(start, start + chunk_size)
        chunk = torch.from_numpy(zyx_np[sl]).to(device)
        w_true[sl], theta_true[sl] = winding_and_theta(phantom_model, chunk)
        if kind == 'model':
            w_cand[sl], theta_cand[sl] = winding_and_theta(payload, chunk)
    if kind == 'volume':
        ij = zyx_np.astype(np.int64)
        w_cand = payload[ij[:, 0], ij[:, 1], ij[:, 2]].astype(np.float32)

    cut_ok = cut_distance(theta_true) > cut_margin
    if theta_cand is not None:
        cut_ok &= cut_distance(theta_cand) > cut_margin
    return score(w_true, w_cand, torn, cut_ok)


@torch.no_grad()
def run_self_test(phantom_dir, num_points, cut_margin, seed, device):
    """Sanity-check the harness itself: the phantom scored against itself must
    be ~exact, and against a perturbed copy must be measurably worse."""
    checkpoint, model, mask = load_phantom(phantom_dir, device)
    identical = evaluate(model, mask, ('model', model), num_points, cut_margin,
                         seed, device)

    perturbed = build_model_from_phantom_checkpoint(checkpoint, device)
    with torch.no_grad():
        hr = perturbed.flow_field.flows[1]
        hr.add_(torch.from_numpy(
            np.random.default_rng(seed + 1).standard_normal(tuple(hr.shape))
            .astype(np.float32)).to(hr.device) * 0.005)
    worse = evaluate(model, mask, ('model', perturbed), num_points, cut_margin,
                     seed, device)

    click.echo('--- phantom vs itself (expect ~0) ---')
    print_report(identical)
    click.echo('--- phantom vs perturbed copy (expect > 0) ---')
    print_report(worse)

    ok = (identical['overall']['mae'] < 1e-3
          and worse['overall']['mae'] > identical['overall']['mae'] * 10)
    click.echo(f'self-test {"PASSED" if ok else "FAILED"}')
    return ok


@click.command()
@click.option('--phantom', required=True, type=click.Path(exists=True, file_okay=False),
              help='synth_phantom output directory (the ground truth).')
@click.option('--candidate-checkpoint', default=None,
              type=click.Path(exists=True, dir_okay=False),
              help='Candidate phantom or fit checkpoint to score.')
@click.option('--candidate-winding-volume', default=None,
              type=click.Path(exists=True, dir_okay=False),
              help='Candidate float32 winding TIFF on the phantom voxel grid.')
@click.option('--umbilicus', default=None, type=click.Path(exists=True, dir_okay=False),
              help='umbilicus json for real fit checkpoints (phantom checkpoints '
                   'carry their own).')
@click.option('--num-points', default=200_000, type=int,
              help='Evaluation samples drawn from the phantom mask.')
@click.option('--cut-margin', default=0.1, type=float,
              help='Angular exclusion band around each theta=0 branch cut, radians.')
@click.option('--seed', default=0, type=int)
@click.option('--output', default=None, type=click.Path(dir_okay=False),
              help='Write the full metrics dict as json.')
@click.option('--self-test', is_flag=True,
              help='Ignore candidates; verify the harness on the phantom itself.')
@click.option('--device', default='cpu', help="torch device ('cpu', 'cuda', ...).")
def main(phantom, candidate_checkpoint, candidate_winding_volume, umbilicus,
         num_points, cut_margin, seed, output, self_test, device):
    device = torch.device(device)

    if self_test:
        raise SystemExit(0 if run_self_test(phantom, num_points, cut_margin,
                                            seed, device) else 1)

    if (candidate_checkpoint is None) == (candidate_winding_volume is None):
        raise click.UsageError(
            'pass exactly one of --candidate-checkpoint / --candidate-winding-volume')

    _, phantom_model, mask = load_phantom(phantom, device)
    if candidate_checkpoint is not None:
        candidate = ('model', load_candidate_model(candidate_checkpoint, umbilicus, device))
    else:
        volume = tifffile.imread(candidate_winding_volume)
        if volume.shape != mask.shape:
            raise click.UsageError(
                f'winding volume shape {volume.shape} != phantom shape {mask.shape}')
        candidate = ('volume', volume)

    result = evaluate(phantom_model, mask, candidate, num_points, cut_margin,
                      seed, device)
    print_report(result)
    if output is not None:
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
