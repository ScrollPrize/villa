"""Standalone debug tool: derive the expected spiral-space winding number of a
seed point from the absolute-winding point-collections attached to its patch.

It reuses fit_spiral's data loading (prepare_patches / prepare_point_collections)
and its spiral transform, but does no fitting/training. Given a checkpoint, a
patches directory, one or more pcl json files and a z-range, it:

  1. loads + z-filters all patches and links every pcl point to them exactly as
     fit_spiral does, yielding the same cross-patch / attached pcl set;
  2. for the user-specified patch, picks a valid vertex near the centre of its
     grid as the seed point;
  3. finds every cross-patch pcl with absolute winding annotations
     (metadata.winding_is_absolute) that has a point attached to that patch;
  4. for each such pcl-point, builds a strip of points on the patch grid between
     the seed and the pcl-point at a fixed --step-size in grid/vertex space
     (taking as many steps as needed), transforms the strip to spiral space and unwraps
     its shifted-radii across the theta=0 seam (exactly as the patch
     winding-number losses do), then propagates the pcl-point's known absolute
     winding along the strip to the seed -> one 'vote' for the seed's winding;
  5. collects the votes as a dict {winding_number: [voters]} and writes json.

The strip-between-points + spiral-space unwrap mirrors get_patch_abs_winding_loss
in fit_spiral.py, but instead of pinning every strip sample to the same target it
derives what the seed *should* be from the annotated point (which can differ from
the raw annotation by the theta=0 unwrap adjustment when the strip crosses the
seam or spans some radial distance).

The transform's flow field is shaped by the checkpoint's own z-range; the
--z-range arg only filters patches/pcls (and must lie within the checkpoint's
range, like fit_spiral's resume path). Both are expressed in full-resolution
scan z (divided by downsample_factor internally), matching fit_spiral's
z_begin/z_end constants.

Run from the spiral/ directory, e.g.

    python find_inconsistent_windings.py \
        --checkpoint out/.../checkpoint_fitted.ckpt \
        --patches-dir /path/to/dataset/patches \
        --patch-id <patch-folder-name> \
        --pcl /path/to/abs_windings.json \
        --z-range 10000,13000
"""

import os
import sys
import json

import click
import numpy as np
import torch

# Import fit_spiral as a module so we can install the checkpoint's cfg + the
# patches/pcl/z-range onto its globals before calling its loaders (which read the
# module globals `cfg`, `patches_path`, `pcl_json_paths`, `z_begin`, `z_end`).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_spiral as fs


def _to_py(x):
    """Convert numpy / torch scalars+arrays to plain python for json."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def install_globals(checkpoint, patches_dir, pcl_paths, filter_z_begin_ds, filter_z_end_ds):
    """Point fit_spiral's module globals at the checkpoint's cfg and the
    user-supplied patches/pcls/filter-z-range so its loaders behave identically.
    Returns the checkpoint's (model) downsampled z-range, which shapes the
    transform's flow field independently of the filtering z-range."""
    fs.cfg = dict(checkpoint['cfg'])
    fs.patches_path = patches_dir
    # Attachment is over the verified patch set only, so skip the (slow, unrelated)
    # unverified patches; they don't change the cross-patch / attached pcl set.
    fs.unverified_patches_path = None
    fs.pcl_json_paths = list(pcl_paths)
    fs.z_begin = filter_z_begin_ds
    fs.z_end = filter_z_end_ds
    # We don't compute shell losses; stop prepare_patches from loading the shell.
    fs.shell_losses_enabled = lambda: False
    return int(checkpoint['z_begin']), int(checkpoint['z_end'])


def build_transform(checkpoint, model_z_begin, model_z_end):
    """Reconstruct the slice->spiral transform and dr_per_winding from the
    checkpoint, using fit_spiral's own model class. The flow field is shaped by
    the checkpoint's (model) z-range, not the filtering z-range."""
    device = torch.device('cuda')
    cfg = fs.cfg

    all_zs = np.arange(model_z_begin, model_z_end)
    umbilicus_fn = fs.umbilicus_z_to_yx(fs.downsample_factor)
    umbilicus_zyx = torch.from_numpy(
        np.concatenate([all_zs[:, None], umbilicus_fn(all_zs)], axis=-1).astype(np.float32)
    ).to(device)

    r = cfg['flow_bounds_radius']
    flow_min_corner = torch.tensor([model_z_begin - cfg['flow_bounds_z_margin'], -r, -r], dtype=torch.int64, device=device)
    flow_max_corner = torch.tensor([model_z_end + cfg['flow_bounds_z_margin'], r, r], dtype=torch.int64, device=device)

    model = fs.SpiralAndTransform(
        flow_integration_steps=cfg['num_flow_integration_steps'],
        flow_integration_solver=cfg['flow_integration_solver'],
        umbilicus_zyx=umbilicus_zyx,
        flow_min_corner_zyx=flow_min_corner,
        flow_max_corner_zyx=flow_max_corner,
    )
    model.to(device)
    model.load_state_dict(checkpoint['spiral_and_transform'])
    model.eval()
    # The high-res flow field is stored "pre-scale": at forward time the hr params
    # are multiplied by flow_scales[1], which training ramps and which is NOT part
    # of the state_dict. A fully-fitted checkpoint ends the ramp at its 'final'
    # value, so pin the scale to that to reproduce the saved transform.
    model.flow_field.flow_scales[1] = cfg['flow_field_high_res_lr_scale_final']

    transform = model.get_slice_to_spiral_transform()
    dr_per_winding = model.get_dr_per_winding()
    return transform, dr_per_winding


def pick_centre_seed(patch):
    """Pick a valid grid point near the centre of the patch's vertex grid. We pick
    the valid quad whose min-corner is closest to the grid centre, so the seed is a
    real vertex that lies on a valid quad (usable by ij_to_zyx / strip sampling)."""
    valid_quad = patch.valid_quad_mask.cpu().numpy()
    qi, qj = np.nonzero(valid_quad)
    if len(qi) == 0:
        raise SystemExit('patch has no valid quads')
    H, W = patch.zyxs.shape[:2]
    centre = np.array([(H - 1) / 2.0, (W - 1) / 2.0])
    d2 = (qi - centre[0]) ** 2 + (qj - centre[1]) ** 2
    k = int(np.argmin(d2))
    return np.array([float(qi[k]), float(qj[k])], dtype=np.float32)


def winding_at_points(transform, dr, zyx):
    """Per-point shifted-radius / dr (the model's raw winding number) for an
    (N, 3) scroll-space tensor; no unwrap (each point treated independently)."""
    spiral = transform(zyx)
    _, _, shifted = fs.get_theta_and_radii(spiral[..., 1:], dr)
    return shifted / dr


def strip_winding_delta(transform, dr, patch, anchor_ij, seed_ij, step_size):
    """Build a straight strip in patch-ij space from `anchor_ij` (a pcl-point) to
    `seed_ij` using a fixed `step_size` in patch grid/vertex (ij) space -- taking
    as many steps as needed to span the seed<->pcl distance -- transform it to
    spiral space, unwrap shifted-radii across the theta=0 seam and return the
    winding-number delta seed-minus-anchor, with diagnostics.

    The number of samples is ceil(||seed - anchor|| / step_size) + 1, so the
    spacing between consecutive samples is <= step_size regardless of how far the
    pcl-point is from the seed (longer strips simply take more steps).

    The unwrap (fit_spiral._unwrap_track_shifted_radii) stitches theta=0 seam
    crossings along the ordered strip, so the difference of unwrapped
    shifted-radii is the true radial gap (incl. whole-winding crossings); /dr
    turns it into a winding-number difference.
    """
    device = dr.device
    anchor_ij = torch.as_tensor(anchor_ij, dtype=torch.float32, device=device)
    seed_ij = torch.as_tensor(seed_ij, dtype=torch.float32, device=device)
    dist = float(torch.linalg.norm(seed_ij - anchor_ij).item())
    num_steps = max(1, int(np.ceil(dist / step_size)))
    num_points = num_steps + 1
    t = torch.linspace(0.0, 1.0, num_points, device=device)[:, None]
    ijs = anchor_ij[None] * (1.0 - t) + seed_ij[None] * t  # index 0 = anchor (pcl), last = seed

    zyx_all, valid = patch.ij_to_zyx(ijs)  # (N, 3), (N,)
    num_invalid = int((~valid).sum().item())
    # Drop strip samples that fall on invalid quads, preserving order. The anchor
    # and seed themselves lie on valid quads, so they survive.
    zyx = zyx_all[valid]
    if zyx.shape[0] < 2:
        return None

    spiral = transform(zyx)
    theta, _, shifted = fs.get_theta_and_radii(spiral[..., 1:], dr)
    shifted_uw = fs._unwrap_track_shifted_radii(theta[None], shifted[None], dr)[0]

    r_anchor = shifted_uw[0]
    r_seed = shifted_uw[-1]
    delta_windings = float(((r_seed - r_anchor) / dr).item())
    return {
        'delta_windings': delta_windings,
        'anchor_raw_winding': float((shifted[0] / dr).item()),
        'seed_raw_winding': float((shifted[-1] / dr).item()),
        'strip_ij_distance': dist,
        'strip_step_size': float(step_size),
        'strip_num_points': int(zyx.shape[0]),
        'strip_num_invalid_dropped': num_invalid,
    }


def parse_z_range(z_range):
    lo, hi = (int(v) for v in z_range.split(','))
    if lo >= hi:
        raise click.BadParameter('--z-range must be "zlo,zhi" with zlo < zhi (full-resolution scan z)')
    return lo, hi


@click.command()
@click.option('--checkpoint', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to a checkpoint_*.ckpt saved by fit_spiral.')
@click.option('--patches-dir', required=True, type=click.Path(exists=True, file_okay=False),
              help='Directory of patch tifxyz folders (the full set, for attachment).')
@click.option('--patch-id', required=True, help='Folder name (within --patches-dir) of the patch to debug.')
@click.option('--pcl', 'pcl_paths', required=True, multiple=True, type=click.Path(exists=True, dir_okay=False),
              help='Point-collection json file(s). Repeat --pcl for several.')
@click.option('--z-range', required=True,
              help='Patch/pcl filtering z-range as "zlo,zhi" in full-resolution scan z '
                   '(divided by downsample_factor internally). Must lie within the checkpoint z-range.')
@click.option('--step-size', default=1.0, type=float,
              help='Step size (in tifxyz grid cells) along strips within each patch; the '
                   'strip takes ceil(distance/step) steps to reach the pcl-point from the seed '
                   '(so spacing between samples is <= step-size, independent of strip length).')
@click.option('--output', default=None, type=click.Path(dir_okay=False),
              help='Output json path (default: winding_votes_<patch>.json next to the checkpoint).')
def main(checkpoint, patches_dir, patch_id, pcl_paths, z_range, step_size, output):
    torch.set_grad_enabled(False)
    device = torch.device('cuda')

    if step_size <= 0:
        raise click.BadParameter('--step-size must be > 0 (patch grid/vertex units)')

    z_lo_raw, z_hi_raw = parse_z_range(z_range)
    df = fs.downsample_factor
    filter_z_begin, filter_z_end = z_lo_raw // df, z_hi_raw // df

    print(f'loading checkpoint {checkpoint}')
    ckpt = torch.load(checkpoint, map_location='cpu')
    model_z_begin, model_z_end = install_globals(ckpt, patches_dir, pcl_paths, filter_z_begin, filter_z_end)
    print(f'checkpoint (model) z-range: [{model_z_begin}, {model_z_end}); '
          f'filtering z-range: [{filter_z_begin}, {filter_z_end}) (downsampled, df={df})')
    if not (filter_z_begin >= model_z_begin and filter_z_end <= model_z_end):
        raise SystemExit(
            f'filtering z-range [{filter_z_begin}, {filter_z_end}) extends beyond the checkpoint '
            f'z-range [{model_z_begin}, {model_z_end}); the transform is undefined outside its domain.'
        )

    print('loading + z-filtering patches')
    patches, _unverified, _shell = fs.prepare_patches()
    if patch_id not in patches:
        raise SystemExit(
            f'patch id {patch_id!r} not among {len(patches)} loaded patches (after z-filtering); '
            'check --patch-id / --z-range'
        )
    patch = patches[patch_id]

    print('loading + linking point-collections')
    cross_patch_pcls, _unattached = fs.prepare_point_collections(patches)

    print('rebuilding spiral transform from checkpoint')
    transform, dr = build_transform(ckpt, model_z_begin, model_z_end)
    dr_value = float(dr.item())
    print(f'dr_per_winding = {dr_value:.4f}')

    # --- seed: a valid grid vertex near the centre of the patch ---
    seed_ij = pick_centre_seed(patch)
    seed_zyx_ds, seed_valid = patch.ij_to_zyx(torch.from_numpy(seed_ij).to(device))
    assert bool(seed_valid.item())
    seed_zyx_ds = seed_zyx_ds.cpu().numpy()
    seed_model_winding = float(
        winding_at_points(transform, dr, torch.from_numpy(seed_zyx_ds).to(device)[None])[0].item()
    )
    print(f'patch {patch_id}: grid {tuple(patch.zyxs.shape[:2])}; '
          f'seed at grid-centre vertex ij={seed_ij.tolist()}; model raw winding = {seed_model_winding:.3f}')

    # --- gather absolute-winding pcls attached to this patch and vote ---
    votes = {}
    voter_details = []
    num_abs_pcls_on_patch = 0
    for pid, pcl in cross_patch_pcls.items():
        if not pcl.get('metadata', {}).get('winding_is_absolute', False):
            continue
        points_on_patch = pcl.get('points_by_patch', {}).get(patch_id)
        if not points_on_patch:
            continue
        num_abs_pcls_on_patch += 1
        for p in points_on_patch:
            pcl_winding = float(p['winding_annotation'])
            if not np.isfinite(pcl_winding):
                print(f'  pcl {pid} ({pcl.get("name")!r}) point {p["id"]}: no winding annotation; skipped')
                continue
            pcl_ij = np.array(p['on_patch']['ij'], dtype=np.float32)

            result = strip_winding_delta(transform, dr, patch, pcl_ij, seed_ij, step_size)
            if result is None:
                print(f'  pcl {pid} ({pcl.get("name")!r}) point {p["id"]}: strip had <2 valid points; skipped')
                continue

            expected = pcl_winding + result['delta_windings']
            expected_int = int(np.round(expected))

            voter = {
                'pcl_id': int(pid),
                'pcl_name': pcl.get('name'),
                'source_file': pcl.get('source_file'),
                'point_id': int(p['id']),
                'pcl_winding_annotation': pcl_winding,
                'pcl_ij': pcl_ij.tolist(),
                'pcl_zyx_downsampled': _to_py(p['zyx']),
                'pcl_distance_to_patch': float(p['on_patch'].get('distance', float('nan'))),
                'expected_seed_winding': expected,
                'expected_seed_winding_rounded': expected_int,
                'delta_windings_pcl_to_seed': result['delta_windings'],
                'model_raw_winding_at_pcl_point': result['anchor_raw_winding'],
                'model_raw_winding_at_seed_via_strip': result['seed_raw_winding'],
                'strip_ij_distance': result['strip_ij_distance'],
                'strip_step_size': result['strip_step_size'],
                'strip_num_points': result['strip_num_points'],
                'strip_num_invalid_dropped': result['strip_num_invalid_dropped'],
            }
            voter_details.append(voter)
            votes.setdefault(str(expected_int), []).append(voter)
            print(f'  pcl {pid} ({pcl.get("name")!r}) point {p["id"]}: '
                  f'annotation={pcl_winding:g} -> expected seed winding={expected:.3f} '
                  f'(round {expected_int}); strip delta={result["delta_windings"]:+.3f}')

    agreeing = sorted(votes.keys(), key=lambda k: int(k))
    print(f'\n{num_abs_pcls_on_patch} absolute-winding pcl(s) on patch; '
          f'{len(voter_details)} vote(s) across winding number(s): '
          f'{", ".join(agreeing) if agreeing else "(none)"}')
    if len(agreeing) > 1:
        print('WARNING: votes disagree on the seed winding number!')

    out = {
        'checkpoint': os.path.abspath(checkpoint),
        'patches_dir': os.path.abspath(patches_dir),
        'patch_id': patch_id,
        'pcl_paths': [os.path.abspath(p) for p in pcl_paths],
        'downsample_factor': df,
        'model_z_range_downsampled': [model_z_begin, model_z_end],
        'filter_z_range_downsampled': [filter_z_begin, filter_z_end],
        'seed': {
            'ij': seed_ij.tolist(),
            'zyx_downsampled': seed_zyx_ds.tolist(),
            'xyz_downsampled': [float(seed_zyx_ds[2]), float(seed_zyx_ds[1]), float(seed_zyx_ds[0])],
        },
        'dr_per_winding': dr_value,
        'model_raw_winding_at_seed': seed_model_winding,
        'num_absolute_pcls_on_patch': num_abs_pcls_on_patch,
        'votes_agree': len(agreeing) <= 1,
        'votes': votes,
    }

    if output is None:
        output = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), f'winding_votes_{patch_id}.json')
    with open(output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote {output}')


if __name__ == '__main__':
    main()
