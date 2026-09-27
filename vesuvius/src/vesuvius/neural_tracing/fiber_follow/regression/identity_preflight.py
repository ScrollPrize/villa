"""Checks before an identity run: negative yield per contact, spot-check images, read cost, GPU fit.

  python -m vesuvius.neural_tracing.fiber_follow.regression.identity_preflight --out PREFLIGHT \\
      --contacts output/direct_ct_spatial_run1/contacts.json

Negatives are validated against annotations the sampler never uses: a
negative near another annotated fiber is confirmed; one near the traced
fiber's own annotation would be a labelling error.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

from vesuvius.neural_tracing.fiber_follow.shared.components import ComponentRule
from vesuvius.neural_tracing.fiber_follow.shared.data import (
    SampleConfig, ZBand, load_fibers, make_sample, split_fibers, tight_block, training_state_allowed,
)
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolume, FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.regression.data import (
    IdentityObservationBuilder, IdentitySampling, contact_location, load_contacts, patch_layout,
)
from vesuvius.neural_tracing.fiber_follow.regression.model import IdentityConfig, IdentityFollower, DirectConfig, DirectFollower

DATA = dict(fiber_zarrs='/mnt/raid_nvme/spiral_dataset_working/fiber_zarrs',
            fibers='/mnt/raid_nvme/spiral_dataset_working/fibers',
            ct='/mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr')


def states(builder, fibers, sample, band, locations, rng):
    items = []
    for location in locations:
        fiber = fibers[location['fiber']]
        item = make_sample(fiber, location['t'], location['reverse'], sample, rng)
        item.update(fiber_ref=(location['fiber'], location['t'], location['reverse']), source=0, source_step=-1, stratum=-1,
                    location_source=location['source'], episode=location.get('episode', -1))
        item = builder.prepare(item, fiber, rng)
        if builder.footprint_allowed(item, band) and all(
                training_state_allowed(item, crop, band) for crop in (builder.cfg.fine, builder.cfg.coarse)):
            items.append(item)
    return items


def negative_audit(items, batch, fibers, all_points, all_ids, own_trees):
    """Distances of sampled negatives to annotations that the sampler never used."""
    rows = []
    K = batch['positive_mask'].shape[1]
    tree = cKDTree(all_points)
    for j, item in enumerate(items):
        fi = item['fiber_ref'][0]
        pos, frame = np.asarray(item['pos']), np.asarray(item['frame'])
        neg = batch['identity_points'][j, K:].numpy()[batch['negative_mask'][j].numpy().reshape(-1) > 0]
        scored = int((batch['positive_mask'][j].numpy()*(batch['negative_mask'][j].numpy().sum(-1) > 0)).sum())
        row = dict(episode=item.get('episode', -1), source=item['location_source'], scored_positives=scored,
                   positives=int(batch['positive_mask'][j].sum()), negatives=len(neg),
                   components=int(batch['foreign_components'][j]))
        if len(neg):
            world = pos+neg @ frame.T
            own = own_trees[fi].query(world)[0]
            others = []
            for w in world:
                hits = [i for i in tree.query_ball_point(w, 3.) if all_ids[i] != fi]
                others.append(min((np.linalg.norm(all_points[i]-w) for i in hits), default=np.inf))
            row.update(own_distance=own.tolist(), other_distance=np.asarray(others).tolist())
        rows.append(row)
    return rows


def spot_images(items, batch, cfg, threshold, path, count=12):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    K = batch['positive_mask'].shape[1]
    chosen = [j for j in range(len(items)) if batch['negative_mask'][j].sum() > 0][:count]
    if not chosen:
        return 0
    fig, axes = plt.subplots(len(chosen), 3, figsize=(10.5, 3.4*len(chosen)))
    axes = np.atleast_2d(axes)
    crop = cfg.fine
    extent = (crop.lateral_coords[0], crop.lateral_coords[-1], crop.lateral_coords[-1], crop.lateral_coords[0])
    for row, j in enumerate(chosen):
        k = int(np.flatnonzero((batch['negative_mask'][j].sum(-1) > 0).numpy())[0])
        positive = batch['identity_points'][j, k].numpy()
        negatives = batch['identity_points'][j, K:].reshape(K, -1, 3)[k].numpy()[batch['negative_mask'][j, k].numpy() > 0]
        z = int(round(positive[2]/crop.spacing+crop.behind))
        ct, presence = batch['x']['fine'][j, 0, z].numpy(), batch['x']['fine'][j, 1, z].numpy()
        foreign = batch['foreign'][j, z].numpy()
        curve = items[j]['identity_curve']
        near = curve[np.abs(curve[:, 2]-positive[2]) < .5]
        for ax, image, title in ((axes[row, 0], ct, 'CT'), (axes[row, 1], presence, 'presence')):
            ax.imshow(image, cmap='gray', extent=extent)
            ax.contour(crop.lateral_coords, crop.lateral_coords, presence, levels=[threshold],
                       colors='yellow', linewidths=.6)
            ax.contourf(crop.lateral_coords, crop.lateral_coords, foreign, levels=[.5, 1.5], colors='red', alpha=.25)
            ax.plot(near[:, 0], near[:, 1], 'c.', ms=2, label='own annotation')
            ax.plot(*positive[:2], 'g+', ms=12, mew=2, label='positive')
            ax.plot(negatives[:, 0], negatives[:, 1], 'rx', ms=7, label='negatives')
            ax.set_title(f"{title} c={positive[2]:.1f} src={items[j]['location_source']}", fontsize=8)
        centers = [p for p in range(cfg.recent_patches) if batch['x']['patch_mask'][j, p] > 0][::4][:8]
        mosaic = [batch['x']['patches'][j, p, cfg.patch_crop.behind].numpy() for p in centers]
        if mosaic:
            axes[row, 2].imshow(np.concatenate(mosaic, 1), cmap='gray')
            axes[row, 2].set_title('history patches (centre slice), newest left; on-fiber ' +
                                   ''.join(str(int(batch['patch_on_fiber'][j, p])) for p in centers), fontsize=7)
        axes[row, 2].axis('off')
    axes[0, 0].legend(fontsize=6, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return len(chosen)


def read_cost(builder, items, vol, repeats=3):
    """Main-crop and history-patch read costs, excluding training query I/O."""
    cfg = builder.cfg
    crops = dict(fine=[], coarse=[], patches=[])
    for _ in range(repeats):
        started = time.perf_counter()
        super(IdentityObservationBuilder, builder).images(items, vol)
        crops['fine'].append(time.perf_counter()-started)
        started = time.perf_counter()
        builder.patch_inputs(items, vol)
        crops['patches'].append(time.perf_counter()-started)
    voxels = dict(fine=0, patches=0, patch_count=0)
    for item in items:
        patch_layout(item, cfg)
        voxels['fine'] += int(np.prod(tight_block(item['pos'], item['frame'], cfg.fine, vol.input_scale)[1]))
        for p in np.flatnonzero(item['patch_mask']):
            frame = np.asarray(item['frame'])
            size = tight_block(item['pos']+frame @ item['patch_centers'][p], frame @ item['patch_frames'][p],
                               cfg.patch_crop, vol.input_scale)[1]
            voxels['patches'] += int(np.prod(size))
            voxels['patch_count'] += 1
    n = len(items)
    return dict(states=n, fine_and_coarse_seconds_per_state=float(np.median(crops['fine'])/n),
                patch_seconds_per_state=float(np.median(crops['patches'])/n),
                fine_block_voxels_per_state=voxels['fine']/n, patch_voxels_per_state=voxels['patches']/n,
                patches_per_state=voxels['patch_count']/n)


def gpu_check(builder, items, vol, microbatch, steps=3):
    """Peak memory and time of full-size identity updates, and single-decision latency."""
    import copy
    from vesuvius.neural_tracing.fiber_follow.regression.train import conv_memory_format, move_batch, optimizer_update
    device = 'cuda'
    torch.manual_seed(0)
    model = IdentityFollower(builder.cfg).to(device, memory_format=conv_memory_format(device))
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = builder(items[:microbatch], vol)
    for key in ('source', 'source_step', 'stratum'):
        batch[key] = torch.zeros(len(items[:microbatch]))
    batch = {k: (v.pin_memory() if torch.is_tensor(v) else {a: b.pin_memory() for a, b in v.items()}) for k, v in batch.items()}
    torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(1, steps+1):
        torch.cuda.synchronize(); started = time.perf_counter()
        metrics = optimizer_update(model, ema, opt, [batch], step, 1e-4, device=device, compute_metrics=step == steps)
        torch.cuda.synchronize(); times.append(time.perf_counter()-started)
    result = dict(microbatch=microbatch, update_seconds=times, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  parameters=sum(p.numel() for p in model.parameters()), loss=metrics['loss'],
                  identity=metrics.get('identity'))
    latency = {}
    one = move_batch({k: v for k, v in builder(items[:1], vol).items() if k in ('x', 'hist', 'hmask')}, device)
    baseline = DirectFollower(DirectConfig()).to(device, memory_format=conv_memory_format(device)).eval()
    for name, net, x in (('identity', model.eval(), one['x']),
                         ('direct', baseline, {k: one['x'][k] for k in ('fine', 'coarse')})):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(3):
                net(x, one['hist'], one['hmask'])
            torch.cuda.synchronize(); started = time.perf_counter()
            for _ in range(10):
                net(x, one['hist'], one['hmask'])
            torch.cuda.synchronize()
        latency[name] = (time.perf_counter()-started)/10
    result['decision_forward_seconds'] = latency
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--contacts', required=True)
    ap.add_argument('--negative-bank', required=True)
    ap.add_argument('--fibers', default=DATA['fibers'])
    ap.add_argument('--fiber-zarrs', default=DATA['fiber_zarrs'])
    ap.add_argument('--ct', default=DATA['ct'])
    ap.add_argument('--episodes', type=int, default=150)
    ap.add_argument('--uniform', type=int, default=150, help='Ordinary fresh states for comparison')
    ap.add_argument('--negative-threshold', type=float, default=ComponentRule().threshold)
    ap.add_argument('--identity-query-patches', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--gpu', action=argparse.BooleanOptionalAction, default=torch.cuda.is_available())
    ap.add_argument('--microbatch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    spec = FiberVolumeSpec(args.fiber_zarrs, ct_zarr=args.ct, ct_level=0, ct_grid_scale=4., inputs='ct+presence')
    fibers = load_fibers(args.fibers, grid_scale=spec.grid_scale)
    band = ZBand(45000/spec.grid_scale, 48500/spec.grid_scale)
    train, _ = split_fibers(fibers, band)
    contacts = load_contacts(args.contacts, train, band)
    cfg = IdentityConfig()
    from vesuvius.neural_tracing.fiber_follow.regression.neighbor_bank import NeighborBank
    bank = NeighborBank(args.negative_bank,train,band,grid_scale=spec.grid_scale)
    bank.validate_volume(spec)
    radius = bank.run['mining']['max_distance'] if args.identity_query_patches else ComponentRule().lateral_max
    sampling = IdentitySampling(rule=ComponentRule(threshold=args.negative_threshold,lateral_max=radius),
                                query_patches=args.identity_query_patches)
    builder = IdentityObservationBuilder(cfg, train, sampling, contacts=contacts,negative_bank=bank)
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history, n_future=cfg.n_future, future_step=cfg.future_step,
                          recent_history_points=cfg.n_history, no_history_prob=0., short_history_prob=0.)
    vol = FiberVolume(spec, cache_bytes=2 << 30)
    picked = rng.choice(len(contacts), size=min(args.episodes, len(contacts)), replace=False)
    locations = []
    for e in picked:
        for side in (0, 1):
            location = contact_location(train, contacts[e], side, bool(rng.integers(2)), rng)
            locations.append(dict(location, episode=int(e)))
    lengths = np.array([f.length for f in train])
    for _ in range(args.uniform):
        fi = int(rng.choice(len(train), p=lengths/lengths.sum()))
        locations.append(dict(fiber=fi, t=float(rng.uniform(0, train[fi].length)), reverse=bool(rng.integers(2)), source=0))
    items = states(builder, train, sample, band, locations, rng)
    print(f'{len(items)} of {len(locations)} states outside the holdout', flush=True)
    all_points = np.concatenate([f.points for f in train])
    all_ids = np.concatenate([np.full(len(f.points), i) for i, f in enumerate(train)])
    own_trees = {}
    rows, first = [], None
    started = time.perf_counter()
    for offset in range(0, len(items), 16):
        chunk = items[offset:offset+16]
        batch = builder(chunk, vol)
        for item in chunk:
            own_trees.setdefault(item['fiber_ref'][0], cKDTree(train[item['fiber_ref'][0]].points))
        rows.extend(negative_audit(chunk, batch, train, all_points, all_ids, own_trees))
        if first is None and batch['negative_mask'].sum() > 0:
            first = (chunk, batch)
        if offset % 64 == 0:
            print(f'  audited {offset+len(chunk)} states, {time.perf_counter()-started:.0f}s', flush=True)
    images = spot_images(*first, cfg, args.negative_threshold, args.out/'negatives.png') if first else 0
    report = dict(threshold=args.negative_threshold, rule=sampling.rule.__dict__, states=len(items), images=images)
    for name, members in (('contact', [r for r in rows if r['source'] == 1]), ('uniform', [r for r in rows if r['source'] == 0])):
        own = np.concatenate([r.get('own_distance', []) for r in members]) if members else np.zeros(0)
        other = np.concatenate([r.get('other_distance', []) for r in members]) if members else np.zeros(0)
        report[name] = dict(
            states=len(members),
            states_with_scored_positive=float(np.mean([r['scored_positives'] > 0 for r in members])) if members else None,
            scored_positives_per_state=float(np.mean([r['scored_positives'] for r in members])) if members else None,
            negatives_per_state=float(np.mean([r['negatives'] for r in members])) if members else None,
            negatives=int(len(own)),
            negative_within_1_5_of_own_annotation=float(np.mean(own <= 1.5)) if len(own) else None,
            negative_within_2_of_other_annotation=float(np.mean(other <= 2.)) if len(other) else None,
            negative_within_3_of_other_annotation=float(np.mean(other <= 3.)) if len(other) else None)
    episodes = {}
    for r in rows:
        if r['episode'] >= 0:
            episodes.setdefault(r['episode'], []).append(r['scored_positives'] > 0)
    yields = np.array([np.mean(v) for v in episodes.values()])
    report['per_episode_yield'] = dict(episodes=len(yields), any_state=float(np.mean(yields > 0)) if len(yields) else None,
                                       quantiles=np.quantile(yields, [.1, .25, .5, .75, .9]).tolist() if len(yields) else None)
    report['read_cost'] = read_cost(builder, items[:16], vol)
    if args.gpu:
        report['gpu'] = gpu_check(builder, items, vol, args.microbatch)
    (args.out/'report.json').write_text(json.dumps(report, indent=2))
    (args.out/'negative_rows.json').write_text(json.dumps(rows))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
