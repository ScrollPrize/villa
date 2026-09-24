"""Cross-component pin pair breakdown of a pinned-fit checkpoint.

    python tests/diag_pin_pairs.py --dataset <root> --checkpoint <run>/checkpoint.pt \
        [--cache <dir>] [--dump pairs.npz]

Loads the checkpoint the way a resume does, reinstalls the full (undemoted)
pin registry and reports, over every 4th patch pin, the cross-component pairs
within 30 voxels that the pinned map cannot honour, split by cause:

  - same slot, different sheet: equal integer targets, free-map winding
    difference over half a winding;
  - different slot, inverted: the free map orders the pair against its
    integer targets;
  - different slot, compressed: ordered correctly but squeezed below 1/16 of
    the target spacing (pins.conflicting_patch_demotion's minimum rise).

and how many of those involve a patch the fit has demoted (the last entry of
the run's pin_demotion.jsonl, else the registry's excluded patches), plus the
modular residual the pair-agreement loss acts on and each pin's distance from
its component target. ``--dump`` writes the pairs for further analysis. The
numbers are the ones pinned_spiral_status.md quotes as "conflicts".
"""

import argparse
import json
import os
import sys

SPIRAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SPIRAL_DIR)


def _demoted_patches(run_dir, registry):
    path = os.path.join(run_dir, 'pin_demotion.jsonl')
    if os.path.exists(path):
        with open(path) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if rows:
            return sorted({entry['patch'] for entry in rows[-1]['demoted']})
    excluded = registry.excluded_patches
    return sorted(excluded.tolist()) if excluded is not None else []


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--scroll-spec', default=None)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache', default=None)
    parser.add_argument('--tolerance', type=float, default=30.0)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--dump', default=None)
    args = parser.parse_args()

    os.environ.setdefault('WANDB_MODE', 'disabled')
    import numpy as np
    import torch
    from scipy.spatial import cKDTree

    torch._C._jit_override_can_fuse_on_gpu(False)
    from config import Config, FitConfig
    from fit_session import conventional_input_paths, default_user_cache_dir, load_scroll_spec
    from fit_spiral import DistributedContext, FitContext, get_env_config_overrides

    scroll_spec = load_scroll_spec(args.dataset, args.scroll_spec)
    paths = conventional_input_paths(args.dataset, scroll_spec)
    config = Config().as_dict()
    config.update(get_env_config_overrides())
    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    ctx = FitContext(FitConfig(config), scroll=scroll_spec, paths=paths, progress=None,
                     resume_path=args.checkpoint,
                     run_dir=os.path.join(run_dir, 'diag_pin_pairs'),
                     cache_dir=args.cache or default_user_cache_dir(),
                     dist_context=DistributedContext.from_env())
    ctx.check_cuda_ready()
    ctx.load_host_inputs()
    ctx.resolve_output_path()
    ctx.build_device_state()
    model = ctx.spiral_and_transform
    if model.pin_registry is None:
        sys.exit('checkpoint has no pin registry (model_pins_enabled off)')
    demoted = _demoted_patches(run_dir, model.pin_registry)
    T = model.effective_pin_targets().detach()
    print(f'pins_active {model.pins_active}, registry pins {model.pin_registry.num_pins}, '
          f'|T - round T| mean {float((T - T.round()).abs().mean()):.3f}')

    with torch.no_grad():
        model.set_pin_registry(ctx._full_pin_registry(), reset_targets=False)
        reg = model.pin_registry
        T = model.effective_pin_targets()
        _, estimate = model.estimate_pin_targets(return_per_pin=True)
        pins_t = model.compute_pins(full=True)
        n = torch.round(pins_t[:, 3] / model.get_dr_per_winding() - T[reg.component])
        dr = float(model.get_dr_per_winding())

    sel = np.arange(0, reg.num_pins, max(args.stride, 1))
    sel = sel[(reg.patch_index[sel] >= 0).cpu().numpy()]
    pts = reg.zyx[sel].cpu().numpy().astype(np.float64)
    pidx = reg.patch_index[sel].cpu().numpy()
    comp = reg.component[sel].cpu().numpy()
    nn = n[sel].cpu().numpy()
    est = estimate[sel].cpu().numpy()
    Tn = T.cpu().numpy()
    pairs = cKDTree(pts).query_pairs(r=args.tolerance, output_type='ndarray')
    a, b = pairs[:, 0], pairs[:, 1]
    cross = comp[a] != comp[b]
    a, b = a[cross], b[cross]
    dS = np.round(Tn[comp[a]] + nn[a]) - np.round(Tn[comp[b]] + nn[b])
    dW = (est[a] + nn[a]) - (est[b] + nn[b])
    same = dS == 0
    inc_same = same & (np.abs(dW) > 0.5)
    inc_inv = ~same & (dW * dS <= 0)
    inc_comp = ~same & (dW * dS > 0) & (np.abs(dW) < np.abs(dS) / 16)
    inc = inc_same | inc_inv | inc_comp
    isdem = np.isin(pidx[a], demoted) | np.isin(pidx[b], demoted)
    pp = ~isdem

    print(f'cross-component pairs {len(a)}, inconsistent {inc.sum()} ({inc.mean():.4f}); '
          f'demoted patches {len(demoted)}')
    print(f'  by cause: same-slot different sheet {inc_same.sum()}, '
          f'different-slot inverted {inc_inv.sum()}, compressed {inc_comp.sum()}')
    print(f'  involving a demoted patch: {(inc & isdem).sum()} '
          f'({(inc & isdem).sum() / max(inc.sum(), 1):.2f}); between pinned patches: {(inc & pp).sum()}')
    if (pp & same).any():
        print(f'  pinned-pinned pairs {pp.sum()}, inconsistent rate {inc[pp].mean():.4f}; '
              f'same-slot {(pp & same).sum()}: |dW| p50/p90/p99 '
              f'{np.round(np.quantile(np.abs(dW[pp & same]), [.5, .9, .99]), 3).tolist()}, '
              f'frac > 0.5 {inc_same[pp].sum() / max((pp & same).sum(), 1):.4f}')
    if (pp & ~same).any():
        print(f'  pinned-pinned different-slot pairs {(pp & ~same).sum()}: inverted rate '
              f'{inc_inv[pp].sum() / max((pp & ~same).sum(), 1):.4f}; '
              f'|dW| median {np.median(np.abs(dW[pp & ~same])):.3f}; '
              f'dS = 1 share {(np.abs(dS[pp & ~same]) == 1).mean():.2f}')
    if len(a):
        mod = np.abs(dW - np.round(dW))
        print(f'  modular residual |dW - round dW| p50/p75/p90/p99 '
              f'{np.round(np.quantile(mod, [.5, .75, .9, .99]), 3).tolist()}, '
              f'frac > 0.05 {(mod > 0.05).mean():.3f}, frac > 0.25 {(mod > 0.25).mean():.3f}')
    strain = np.abs(est - Tn[comp])
    in_demoted = np.isin(pidx, demoted)
    print(f'  |estimate - T| per pin: pinned patches median {np.median(strain[~in_demoted]):.3f} '
          f'p90 {np.quantile(strain[~in_demoted], .9):.3f}; demoted patches median '
          f'{np.median(strain[in_demoted]) if in_demoted.any() else float("nan"):.3f}')
    if args.dump:
        np.savez(args.dump, pts_a=pts[a].astype(np.float32), pts_b=pts[b].astype(np.float32),
                 dW=dW.astype(np.float32), dS=dS.astype(np.int16),
                 pa=pidx[a].astype(np.int32), pb=pidx[b].astype(np.int32),
                 est_a=est[a].astype(np.float32), est_b=est[b].astype(np.float32),
                 Ta=Tn[comp[a]].astype(np.float32), Tb=Tn[comp[b]].astype(np.float32),
                 na=nn[a].astype(np.int16), nb=nn[b].astype(np.int16), isdem=isdem, dr=dr)


if __name__ == '__main__':
    main()
