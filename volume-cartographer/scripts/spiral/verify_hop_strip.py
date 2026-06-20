"""Debug/verification tool: re-measure every within-patch strip along one vote's
BFS chain from find_inconsistent_windings, with instrumentation, to find which
hop's integer theta=0 seam count is unreliable (the source of an off-by-one vote).

For a chosen target vote (by its rounded expected winding) in a winding_votes.json
produced by find_inconsistent_windings.py, this rebuilds the *same* spiral
transform from the checkpoint and re-runs each hop's within-patch strip
(seed-entry strip for hop 1, each tree edge's intra-patch strip, and the final
anchor->entry strip), reporting per strip:

  * the integer delta_windings (theta=0 seam-crossing count) used for propagation,
  * the continuous unwrapped-shifted-radius residual (a diagnostic),
  * the number of +pi / -pi seam crossings,
  * min_margin_to_pi: how close the closest theta step came to the ambiguous +-pi
    boundary (a small margin means the crossing count is a coin-flip), and
  * whether the integer delta_windings stays stable under a sweep of --step-size
    and --medial-weight (an unstable hop is the off-by-one culprit).

Read-only: loads the checkpoint + patches + pcls and computes; writes nothing.

Run from scripts/spiral/ with the SAME --checkpoint/--patches-dir/--pcl/--z-range
as the original find_inconsistent_windings run, plus --votes-json/--target-winding.
"""

import json

import click
import numpy as np
import torch

import fit_spiral as fs
import connect_overlapping_patches as cop
import find_inconsistent_windings as fiw


def instrumented_strip(transform, dr, graph, from_ij, to_ij, step_size):
    """Mirror of fiw.strip_winding_delta but also returns the theta trace + the
    seam-crossing diagnostics. Returns None if the strip can't be measured."""
    device = dr.device
    from_ij = np.asarray(from_ij, dtype=np.float32)
    to_ij = np.asarray(to_ij, dtype=np.float32)

    centres = fiw._valid_quad_path_centres(
        graph,
        fiw._nearest_valid_quad_node(graph, from_ij),
        fiw._nearest_valid_quad_node(graph, to_ij),
    )
    if centres is None:
        return None

    polyline = [from_ij] + [np.array(c, dtype=np.float32) for c in centres] + [to_ij]
    ijs = torch.from_numpy(fiw._polyline_ijs(polyline, step_size)).to(device)

    zyx_all, valid = graph.patch.ij_to_zyx(ijs)
    zyx = zyx_all[valid]
    if zyx.shape[0] < 2:
        return None

    spiral = transform(zyx)
    theta, _, shifted = fs.get_theta_and_radii(spiral[..., 1:], dr)
    theta = theta.detach()
    theta_diffs = theta[1:] - theta[:-1]
    pos_cross = (theta_diffs > np.pi)
    neg_cross = (theta_diffs < -np.pi)
    cumulative = (pos_cross.to(shifted.dtype) - neg_cross.to(shifted.dtype)).sum()
    delta_windings = int(round(float((-cumulative).item())))

    shifted_uw = fs._unwrap_track_shifted_radii(theta[None], shifted[None], dr)[0]
    residual = float(((shifted_uw[-1] - shifted_uw[0]) / dr).item())

    abs_diffs = theta_diffs.abs()
    # How close the closest step came to the +-pi seam (min over steps of |pi - |dtheta||);
    # a near-zero margin means a borderline crossing that could be counted either way.
    margins = (np.pi - abs_diffs).abs()
    min_margin = float(margins.min().item())
    min_margin_idx = int(margins.argmin().item())
    return {
        'delta_windings': delta_windings,
        'residual': residual,
        'num_points': int(zyx.shape[0]),
        'num_quads': len(centres),
        'num_pos_cross': int(pos_cross.sum().item()),
        'num_neg_cross': int(neg_cross.sum().item()),
        'min_margin_to_pi': min_margin,
        'min_margin_step_idx': min_margin_idx,
        'max_abs_theta_step': float(abs_diffs.max().item()),
    }


def collect_strips(votes, target_winding, seed_ij):
    """From the chosen vote, build the ordered list of strips to re-measure:
    (label, patch_id, from_ij, to_ij). Hop 1's strip starts at the seed entry; each
    later hop's strip starts at the previous hop's arrival; the final strip is the
    anchor->entry strip on the anchor patch."""
    cands = votes.get(str(int(target_winding)), [])
    if not cands:
        raise SystemExit(f'no vote rounding to {target_winding} in votes json')
    if len(cands) > 1:
        print(f'note: {len(cands)} votes round to {target_winding}; using the deepest (most hops)')
    v = max(cands, key=lambda c: c['hops'])
    path = v['path']

    strips = []
    entry = np.asarray(seed_ij, dtype=np.float32)  # entry point on the current patch
    for i, h in enumerate(path):
        # within-patch strip on h['from_patch'] from the current entry to the edge departure
        strips.append((f'hop{i + 1} {h["rel_pcl_name"]} on {h["from_patch"]}',
                       h['from_patch'], entry, np.asarray(h['from_ij'], dtype=np.float32)))
        entry = np.asarray(h['to_ij'], dtype=np.float32)  # arrival becomes next patch's entry
    # final anchor->entry strip on the anchor patch
    anchor_patch = v['abs_patch_id']
    strips.append((f'anchor {v["abs_pcl_name"]} pt{v["abs_point_id"]} on {anchor_patch}',
                   anchor_patch, np.asarray(v['abs_ij'], dtype=np.float32), entry))
    return v, strips


@click.command()
@click.option('--checkpoint', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--patches-dir', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--umbilicus', default=None, type=click.Path(exists=True, dir_okay=False))
@click.option('--pcl', 'pcl_paths', required=True, multiple=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--z-range', required=True)
@click.option('--votes-json', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--target-winding', required=True, type=int,
              help='Rounded expected winding of the vote whose chain to re-measure (e.g. 19).')
@click.option('--step-size', default=1.0, type=float, help='Base sampling spacing.')
@click.option('--medial-weight', default=4.0, type=float, help='Base medial weighting.')
@click.option('--sweep/--no-sweep', default=True,
              help='Re-measure each strip over step-size {0.5,1,2} x medial-weight {0,4,8} and '
                   'flag any strip whose integer delta is unstable.')
def main(checkpoint, patches_dir, umbilicus, pcl_paths, z_range, votes_json, target_winding,
         step_size, medial_weight, sweep):
    torch.set_grad_enabled(False)

    z_lo, z_hi = fiw.parse_z_range(z_range)
    df = fs.downsample_factor
    filter_z_begin, filter_z_end = z_lo // df, z_hi // df
    umbilicus_path = fiw.resolve_umbilicus_path(patches_dir, umbilicus)

    print(f'loading checkpoint {checkpoint}')
    ckpt = torch.load(checkpoint, map_location='cpu')
    model_z_begin, model_z_end = fiw.install_globals(
        ckpt, patches_dir, pcl_paths, filter_z_begin, filter_z_end, umbilicus_path)
    print('loading + z-filtering patches')
    patches, _u, _s = fs.prepare_patches()
    print('rebuilding spiral transform from checkpoint')
    transform, dr = fiw.build_transform(ckpt, model_z_begin, model_z_end)
    print(f'dr_per_winding = {float(dr.item()):.4f}')

    with open(votes_json) as f:
        vj = json.load(f)
    seed_ij = vj['seed']['ij']
    vote, strips = collect_strips(vj['votes'], target_winding, seed_ij)
    print(f"\ntarget vote: pcl {vote['abs_pcl_id']} ({vote['abs_pcl_name']!r}) pt {vote['abs_point_id']} "
          f"on {vote['abs_patch_id']}, {vote['hops']} hops, "
          f"expected {vote['expected_seed_winding']} (round {vote['expected_seed_winding_rounded']})")

    graph_cache = {}

    def graph_for(pid, mw):
        key = (pid, mw)
        g = graph_cache.get(key)
        if g is None:
            g = cop.build_patch_graph(patches[pid], mw)
            g._dij_cache = {}
            graph_cache[key] = g
        return g

    base_step, base_mw = step_size, medial_weight
    sweep_steps = sorted({0.5, 1.0, 2.0, base_step})
    sweep_mws = sorted({0.0, 4.0, 8.0, base_mw})

    print(f'\n{"strip":<60} {"int":>4} {"resid":>8} {"+x/-x":>7} {"margin_pi":>9} {"pts":>6} {"stable?":>8}')
    print('-' * 110)
    for label, pid, a, b in strips:
        if pid not in patches:
            print(f'{label:<60}  (patch not loaded)')
            continue
        base = instrumented_strip(transform, dr, graph_for(pid, base_mw), a, b, base_step)
        if base is None:
            print(f'{label:<60}  (unmeasurable)')
            continue

        stable = 'n/a'
        deltas = {base['delta_windings']}
        if sweep:
            for ss in sweep_steps:
                for mw in sweep_mws:
                    r = instrumented_strip(transform, dr, graph_for(pid, mw), a, b, ss)
                    if r is not None:
                        deltas.add(r['delta_windings'])
            stable = 'YES' if len(deltas) == 1 else f'NO {sorted(deltas)}'

        flag = ''
        if base['min_margin_to_pi'] < 0.5:
            flag = '  <-- borderline seam (margin<0.5)'
        if sweep and len(deltas) > 1:
            flag = f'  <-- UNSTABLE integer {sorted(deltas)}'
        print(f'{label:<60} {base["delta_windings"]:>+4d} {base["residual"]:>+8.3f} '
              f'{base["num_pos_cross"]}/{base["num_neg_cross"]:<5} {base["min_margin_to_pi"]:>9.4f} '
              f'{base["num_points"]:>6} {stable:>8}{flag}')

    print('\nInterpretation: a strip flagged UNSTABLE (integer flips under the step/medial sweep) '
          'or with a tiny margin_pi is the unreliable seam-count -- the source of the off-by-one '
          'vote. A stable integer with a comfortable margin is trustworthy.')


if __name__ == '__main__':
    main()
