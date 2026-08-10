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

Relation to spiralcheck (github.com/Nicodol/spiralcheck): complementary, not
competing. spiralcheck scores REAL whole-scroll fits from their output meshes
against held-out human-verified patches -- evaluation this tool cannot do,
since real scrolls have no truth. This tool scores against EXACT synthetic
truth at every voxel, including regions no human annotated -- evaluation
held-out patches cannot do, since they inherit human labels' locations and
noise. Use spiralcheck on real fits; use this on phantoms.

Candidates:
  --candidate-checkpoint      a phantom checkpoint (self-tests, method dev) or
                              a real fit checkpoint (needs --umbilicus, loaded
                              via fit_spiral's own umbilicus reader)
  --candidate-winding-volume  a float32 TIFF of per-voxel winding on the
                              phantom's voxel grid -- so ANY method that
                              produces a winding field (e.g. a lasagna winding
                              volume) can be scored, not just spiral fits
  --candidate-tifxyz          a tifxyz surface directory (or a directory of
                              them) -- the pipeline's lingua franca, and what
                              spiralcheck consumes -- scored by where its
                              vertices actually sit in phantom truth

Gauge: a candidate fit of the same volume can differ from the phantom by a
constant winding offset (theta-origin / winding-index choice), so residuals
are aligned by their median before scoring; raw (unaligned) MAE is also
reported. KNOWN CONSEQUENCE: a globally constant mis-count (every voxel
off by exactly +1 winding) is absorbed as gauge and scores as zero error --
inspect the reported gauge offset, which is where such errors surface.
Opposite-sense (CW vs ACW) candidates are not auto-detected.

Topology caveat: switch rate is a POINTWISE assignment metric, not a
connectivity metric -- it does not detect all topological defects (the
surface-detection Kaggle competition used deliberately topology-aware scoring
for that reason, and spiralcheck's intrinsic winding-order checks cover
related ground). The tifxyz mode's grid-discontinuity figure is a first
topology-flavoured check, not a substitute.

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
from tifxyz import load_tifxyz


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


def load_finite_paint_truth(npz_path):
    """Ground truth from a Diego-dcv finite_paint .npz (the consume path):
    turn_id is int16 [Z,Y,X], 0=air, winding t stored as t+1. Returns
    (winding_truth float32 [Z,Y,X] with 0-based winding on material voxels,
    mask bool [Z,Y,X] = material). No transform is involved -- the raster IS
    the truth, so our ruler becomes source-agnostic (any painter that emits a
    per-voxel instance-id volume can be scored, not just synth_phantom).

    NOTE: turn_id is a per-SHEET instance label (constant across a sheet's
    finite thickness), so the scoring below reports switch rate -- whether a
    candidate assigns each material voxel to the correct integer winding -- as
    the primary metric; the continuous winding MAE carries an inherent
    discretisation floor against integer labels.
    """
    data = np.load(npz_path)
    if 'turn_id' not in data:
        raise click.UsageError(
            f'{npz_path} has no turn_id array (keys: {list(data.keys())}); '
            'not a finite_paint-format phantom')
    turn_id = data['turn_id']
    if turn_id.ndim != 3:
        raise click.UsageError(f'turn_id must be 3D [Z,Y,X], got {turn_id.shape}')
    mask = turn_id > 0
    winding = np.where(mask, turn_id.astype(np.float32) - 1.0, np.nan)
    return winding, mask


def score_turn_id(truth_winding, mask, candidate_winding, num_points, rng):
    """Gauge-aligned instance-assignment scoring of a candidate winding raster
    against per-voxel integer truth. switch = wrong-turn fraction (the metric
    that matters for instance labels); mae/raw_mae are the continuous residual."""
    idx = np.stack(np.nonzero(mask), axis=-1)
    if len(idx) > num_points:
        idx = idx[rng.integers(0, len(idx), size=num_points)]
    z, y, x = idx[:, 0], idx[:, 1], idx[:, 2]
    residual = candidate_winding[z, y, x] - truth_winding[z, y, x]
    offset = float(np.median(residual))
    aligned = residual - offset
    per_turn = {}
    turns = truth_winding[z, y, x].astype(np.int64)
    for tval in np.unique(turns):
        r = aligned[turns == tval]
        per_turn[int(tval)] = {'n': int(len(r)), 'mae': float(np.abs(r).mean()),
                               'switch_rate': float((np.abs(r) > 0.5).mean())}
    return {
        'source': 'finite_paint turn_id',
        'gauge_offset': offset,
        'raw_mae': float(np.abs(residual).mean()),
        'overall': {'n': int(len(aligned)), 'mae': float(np.abs(aligned).mean()),
                    'rmse': float(np.sqrt((aligned ** 2).mean())),
                    'switch_rate': float((np.abs(aligned) > 0.5).mean())},
        'per_turn': per_turn,
    }


def print_turn_id_report(result):
    o = result['overall']
    click.echo(f"scored against {result['source']}; gauge {result['gauge_offset']:+.3f}")
    click.echo(f"overall  n={o['n']}  MAE {o['mae']:.4f}  RMSE {o['rmse']:.4f}  "
               f"switch {o['switch_rate'] * 100:.2f}%   (raw MAE {result['raw_mae']:.4f})")
    click.echo('per-turn switch%:')
    for t, s in sorted(result['per_turn'].items()):
        click.echo(f'  turn {t:3d}  {s["switch_rate"] * 100:6.2f}%  (n={s["n"]})')


def load_tifxyz_candidates(path):
    """A single tifxyz directory (contains x.tif) or a directory of them."""
    if os.path.exists(os.path.join(path, 'x.tif')):
        entries = [path]
    else:
        entries = sorted(
            os.path.join(path, e) for e in os.listdir(path)
            if os.path.exists(os.path.join(path, e, 'x.tif')))
    if not entries:
        raise click.UsageError(f'{path} contains no tifxyz surfaces')
    return [(os.path.basename(e), load_tifxyz(e)) for e in entries]


@torch.no_grad()
def evaluate_tifxyz(phantom_model, patches, cut_margin):
    """Score tifxyz surfaces by where their vertices sit in phantom truth.

    on_surface: |w_true - round(w_true)| per valid vertex -- distance from ANY
      true sheet, in winding units. Zero for a perfect surface regardless of
      which winding it traces, so it needs no gauge.
    grid_discontinuity: fraction of grid-adjacent valid vertex pairs whose
      TRUE windings differ by > 0.5 -- the mesh jumps sheets between
      neighbouring grid cells. Pairs straddling the theta=0 cut are excluded
      (w_true legitimately jumps by 1 there on a continuous spiral surface).
    winding_agreement: only when the surface carries per-vertex winding.tif
      annotations -- gauge-aligned agreement with truth, as for other modes.
    """
    per_patch = {}
    for name, patch in patches:
        zyxs = patch.zyxs.reshape(-1, 3)
        valid = patch.valid_vertex_mask.reshape(-1).numpy()
        w_true, theta = winding_and_theta(phantom_model, zyxs)
        on_surface = np.abs(w_true - np.round(w_true))[valid]

        h, w = patch.zyxs.shape[:2]
        w_grid = w_true.reshape(h, w)
        theta_grid = theta.reshape(h, w)
        valid_grid = valid.reshape(h, w)
        pair_jumps = []
        for da, db in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                       ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
            both = valid_grid[da] & valid_grid[db]
            not_seam = np.abs(theta_grid[da] - theta_grid[db]) < np.pi
            sel = both & not_seam
            pair_jumps.append(np.abs(w_grid[da] - w_grid[db])[sel] > 0.5)
        jumps = np.concatenate(pair_jumps)

        agreement = None
        if isinstance(patch.winding, torch.Tensor):
            annotated = patch.winding.reshape(-1).numpy()[valid]
            cut_ok = cut_distance(theta[valid]) > cut_margin
            residual = (w_true[valid] - annotated)[cut_ok]
            offset = float(np.median(residual))
            aligned = residual - offset
            agreement = {'gauge_offset': offset,
                         'mae': float(np.abs(aligned).mean()),
                         'switch_rate': float((np.abs(aligned) > 0.5).mean())}
        per_patch[name] = {
            'num_valid_vertices': int(valid.sum()),
            'on_surface_mae': float(on_surface.mean()),
            'on_surface_p95': float(np.percentile(on_surface, 95)),
            'grid_discontinuity_frac': float(jumps.mean()) if len(jumps) else 0.,
            'winding_agreement': agreement,
        }
    total = sum(p['num_valid_vertices'] for p in per_patch.values())
    overall = {
        'num_patches': len(per_patch),
        'num_valid_vertices': total,
        'on_surface_mae': float(sum(
            p['on_surface_mae'] * p['num_valid_vertices'] for p in per_patch.values()) / total),
        'grid_discontinuity_frac': float(np.mean(
            [p['grid_discontinuity_frac'] for p in per_patch.values()])),
    }
    return {'overall': overall, 'per_patch': per_patch}


def print_tifxyz_report(result):
    o = result['overall']
    click.echo(f"{o['num_patches']} surfaces, {o['num_valid_vertices']} valid vertices")
    click.echo(f"on-surface MAE {o['on_surface_mae']:.4f} windings; "
               f"grid discontinuity {o['grid_discontinuity_frac'] * 100:.2f}% of pairs")
    for name, p in sorted(result['per_patch'].items()):
        agreement = p['winding_agreement']
        suffix = (f"  agreement MAE {agreement['mae']:.4f} "
                  f"(gauge {agreement['gauge_offset']:+.3f})" if agreement else '')
        click.echo(f"  {name}: on-surface {p['on_surface_mae']:.4f}, "
                   f"p95 {p['on_surface_p95']:.4f}, "
                   f"discont {p['grid_discontinuity_frac'] * 100:.2f}%{suffix}")


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
@click.option('--phantom', default=None, type=click.Path(exists=True, file_okay=False),
              help='synth_phantom output directory (the ground truth). Omit when '
                   'using --truth-npz.')
@click.option('--truth-npz', default=None, type=click.Path(exists=True, dir_okay=False),
              help='finite_paint-format .npz (turn_id raster) as ground truth, '
                   'scored against --candidate-winding-volume on the same grid. '
                   'The source-agnostic consume path.')
@click.option('--candidate-checkpoint', default=None,
              type=click.Path(exists=True, dir_okay=False),
              help='Candidate phantom or fit checkpoint to score.')
@click.option('--candidate-winding-volume', default=None,
              type=click.Path(exists=True, dir_okay=False),
              help='Candidate float32 winding TIFF on the phantom voxel grid.')
@click.option('--candidate-tifxyz', default=None,
              type=click.Path(exists=True, file_okay=False),
              help='Candidate tifxyz surface directory (or directory of them).')
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
def main(phantom, truth_npz, candidate_checkpoint, candidate_winding_volume,
         candidate_tifxyz, umbilicus, num_points, cut_margin, seed, output,
         self_test, device):
    device = torch.device(device)

    if self_test:
        if phantom is None:
            raise click.UsageError('--self-test needs --phantom')
        raise SystemExit(0 if run_self_test(phantom, num_points, cut_margin,
                                            seed, device) else 1)

    given = [c for c in (candidate_checkpoint, candidate_winding_volume,
                         candidate_tifxyz) if c is not None]
    if len(given) != 1:
        raise click.UsageError(
            'pass exactly one of --candidate-checkpoint / '
            '--candidate-winding-volume / --candidate-tifxyz')

    # Consume path: external finite_paint turn_id truth, no synth_phantom model.
    if truth_npz is not None:
        if candidate_winding_volume is None:
            raise click.UsageError(
                '--truth-npz scores against --candidate-winding-volume')
        truth_winding, mask = load_finite_paint_truth(truth_npz)
        candidate = tifffile.imread(candidate_winding_volume)
        if candidate.shape != mask.shape:
            raise click.UsageError(
                f'candidate shape {candidate.shape} != truth shape {mask.shape}')
        result = score_turn_id(truth_winding, mask, candidate, num_points,
                               np.random.default_rng(seed))
        print_turn_id_report(result)
        if output is not None:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2)
        return

    if phantom is None:
        raise click.UsageError('pass --phantom (or --truth-npz)')
    _, phantom_model, mask = load_phantom(phantom, device)
    if candidate_tifxyz is not None:
        result = evaluate_tifxyz(phantom_model,
                                 load_tifxyz_candidates(candidate_tifxyz),
                                 cut_margin)
        print_tifxyz_report(result)
        if output is not None:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2)
        return
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
