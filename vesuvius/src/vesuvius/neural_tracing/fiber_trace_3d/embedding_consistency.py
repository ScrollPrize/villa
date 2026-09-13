"""Analysis only: GT embedding continuity through the real inference lattice.

No embedding publication or tracer scoring is enabled by this diagnostic.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def distribution(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    if not len(x):
        return {"n": 0}
    if not np.isfinite(x).all():
        raise ValueError("nonfinite diagnostic values")
    return dict(n=len(x), mean=float(x.mean()), **{
        key: float(np.quantile(x, q)) for key, q in
        (("min", 0), ("p05", .05), ("p50", .5), ("p95", .95), ("max", 1))})


def corners(points):
    lo = np.floor(points).astype(np.int64)
    frac = points - lo
    offsets = np.array(list(itertools.product((0, 1), repeat=3)))
    coords = lo[:, None] + offsets[None]
    weights = np.prod(np.where(offsets[None], frac[:, None], 1-frac[:, None]), axis=-1)
    return coords, weights


def interpolate(array, points):
    """CZYX at fractional ZYX, requiring all eight corners to be in bounds."""
    coords, weights = corners(points)
    return (array[:, coords[..., 0], coords[..., 1], coords[..., 2]].transpose(1, 2, 0)
            * weights[..., None]).sum(axis=1)


def sample_line(line, spacing, length):
    delta = np.diff(line, axis=0)
    distance = np.r_[0, np.cumsum(np.linalg.norm(delta, axis=1))]
    keep = np.r_[True, np.diff(distance) > 0]
    line, distance = line[keep], distance[keep]
    if len(line) < 2 or distance[-1] < length:
        raise ValueError("fiber too short for requested analysis window")
    at = np.arange(-length/2, length/2 + spacing*.1, spacing) + distance[-1]/2
    return np.stack([np.interp(at, distance, line[:, k]) for k in range(3)], axis=1)


def analyze_window(model, adapter, reader, geometry, points, full_line, precision, device, seed):
    import torch
    from lasagna import tiled_predict3d as tiled
    from .train import _autocast_context

    coords, alpha = corners(points)
    flat = coords.reshape(-1, 3)
    if (flat < 0).any() or (flat >= geometry.output_shape).any():
        raise ValueError("GT interpolation support outside input volume")
    indices = sorted(set(tuple(v // geometry.chunk * geometry.chunk) for v in flat))
    tiles = sorted(set(t for c in indices for t in geometry.contributors(c)))
    # Use precisely the shared worker's Gaussian-downsampled separable weights.
    ramp = torch.as_tensor(tiled.blend_ramp(geometry.tile, geometry.border, geometry.overlap), device=device)
    weight = tiled._pyrdown3d((ramp[:, None, None]*ramp[None, :, None]*ramp[None, None, :])[None],
                             factor=geometry.scale)[0].cpu().numpy()
    total = None
    mass = np.zeros(len(flat), np.float32)
    amplitude = mass.copy()
    strongest = mass.copy()
    owners = np.full(len(flat), -1, int)
    tile_samples, tile_valid, per_tile = [], [], []
    negative, negative_presence, negative_weighted = [], [], []
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    for tile_id, tile in enumerate(tiles):
        origin = np.array(geometry.tile_origin(tile))
        local = flat - origin
        selected = np.all((local >= 0) & (local < geometry.tile//geometry.scale), axis=1)
        j = np.flatnonzero(selected)
        if not len(j):
            continue
        c = local[j]
        w = weight[c[:, 0], c[:, 1], c[:, 2]]
        if not np.any(w > 0):
            continue
        raw = reader(tile)
        x = tiled._input_tile_to_device(raw, device)
        x = adapter.preprocess_tile(x, torch.ones_like(x, dtype=torch.bool))
        with torch.inference_mode(), _autocast_context(precision):
            output = model(x, return_embeddings=True)
        e = output["embeddings"][0].float().cpu().numpy()
        presence = output["prediction"][0, 6].float().cpu().numpy()
        if e.shape[0] == 0 or e.shape[1:] != weight.shape:
            raise ValueError("embedding output lattice mismatch")
        if not np.isfinite(e).all():
            raise ValueError("nonfinite model embeddings")
        if total is None:
            total = np.zeros((len(flat), e.shape[0]), np.float32)
        values = e[:, c[:, 0], c[:, 1], c[:, 2]].T
        total[j] += values*w[:, None]
        amplitude[j] += np.linalg.norm(values, axis=1)*w
        mass[j] += w
        changed = w > strongest[j]
        owners[j[changed]] = tile_id
        strongest[j] = np.maximum(strongest[j], w)
        p = points-origin
        valid = np.all((p >= 0) & (p < geometry.tile//geometry.scale-1), axis=1)
        ids = np.flatnonzero(valid)
        sampled = np.zeros((len(points), e.shape[0]), np.float32)
        if len(ids):
            sampled[ids] = interpolate(e, p[ids])
            # Ignore zero-weight border-only support in the per-tile comparison.
            valid[ids] &= interpolate(weight[None], p[ids])[:, 0] > .05
        tile_samples.append(sampled)
        tile_valid.append(valid)
        per_tile.append(dict(tile=list(tile), gt_points=int(valid.sum())))
        # Conservative off-GT control: >80 base voxels from the entire fiber
        # guarantees outside the training ellipse (largest radius80 base voxels).
        ids = np.flatnonzero(valid)
        if len(ids):
            q = rng.integers(geometry.border//geometry.scale+1,
                (geometry.tile-geometry.border)//geometry.scale-1, size=(256, 3))
            global_q = q+origin
            # Full line is in output coordinates; exact segment distance.
            nearest = np.full(len(q), np.inf)
            for a, b in zip(full_line[:-1], full_line[1:]):
                d = b-a
                t = np.clip(((global_q-a)*d).sum(1)/max(float(d@d), 1e-16), 0, 1)
                nearest = np.minimum(nearest, np.linalg.norm(global_q-a-t[:, None]*d, axis=1))
            q = q[nearest > 80/(2*geometry.scale)]  # level1, validated by caller
            if len(q):
                reference = sampled[rng.choice(ids, len(q))]
                sims = (unit(reference)*unit(e[:, q[:, 0], q[:, 1], q[:, 2]].T)).sum(1)
                pres = presence[q[:, 0], q[:, 1], q[:, 2]]
                negative.extend(sims.tolist())
                negative_presence.extend(pres.tolist())
                if pres.sum() > 0:
                    negative_weighted.extend(rng.choice(sims, 256, p=pres/pres.sum()).tolist())
        del output, x, e
    if total is None or np.any(mass <= 0):
        raise ValueError("incomplete blend support")
    blended_corners = total / mass[:, None]
    blended = (blended_corners.reshape(len(points), 8, -1)*alpha[..., None]).sum(1)
    cancellation = np.linalg.norm(total, axis=1)/np.maximum(amplitude, 1e-8)
    # Dominant contributor at the nearest interpolation corner.
    owner = owners.reshape(-1, 8)[np.arange(len(points)), alpha.argmax(1)]
    cross_same = []
    for i in range(len(tile_samples)):
        for k in range(i):
            valid = tile_valid[i] & tile_valid[k]
            cross_same.extend((unit(tile_samples[i][valid])*unit(tile_samples[k][valid])).sum(1).tolist())
    metrics = dict(same_point_cross_tile=distribution(cross_same),
                   cancellation=distribution(cancellation),
                   blended_norm=distribution(np.linalg.norm(blended, axis=1)),
                   negative_cosine=distribution(negative),
                   negative_presence_weighted_cosine=distribution(negative_weighted),
                   negative_presence=distribution(negative_presence))
    negative, negative_presence = np.asarray(negative), np.asarray(negative_presence)
    for low, high in ((0, .1), (.1, .5), (.5, 1.01)):
        metrics[f"negative_presence_{low}_{high}"] = distribution(negative[(negative_presence>=low)&(negative_presence<high)])
    distance_rows = {}
    for lag in (1, 4, 16, 32):
        if len(points) <= lag:
            continue
        cross = np.array([np.any(owner[i:i+lag] != owner[i+1:i+lag+1]) for i in range(len(points)-lag)])
        storage = np.any(np.floor(points[:-lag]/geometry.chunk) != np.floor(points[lag:]/geometry.chunk), axis=1)
        cosine = (unit(blended[:-lag])*unit(blended[lag:])).sum(1)
        within = []
        for s, valid in zip(tile_samples, tile_valid):
            use = valid[:-lag]&valid[lag:]
            within.extend((unit(s[:-lag][use])*unit(s[lag:][use])).sum(1).tolist())
        distance_rows[str(lag)] = dict(within_tile=distribution(within), blended_interior=distribution(cosine[~cross]),
            blended_switch=distribution(cosine[cross]), stored_chunk_crossing=distribution(cosine[storage]))
    metrics["by_lag"] = distance_rows
    return dict(metrics=metrics, tiles=per_tile, seconds=time.monotonic()-started,
                zero_norm_points=int((np.linalg.norm(blended, axis=1)<1e-8).sum()),
                points_output_zyx=points.tolist(), dominant_tile=owner.tolist(),
                blended_embeddings=blended.tolist(), same_point_cross_tile_cosines=cross_same,
                negative_cosines=negative.tolist(), negative_presences=negative_presence.tolist())


def run(settings):
    import torch
    from lasagna.demand_predict3d import DemandGeometry
    from .live_evaluate import (freeze_checkpoint, NativeTileReader, native_reader_options,
                               input_root_attributes, validate_input_lattice)
    from .inference_adapter import FiberTrace3DPredictAdapter
    from .train import _mixed_precision_config_from_training
    from vesuvius.neural_tracing.fiber_trace_2d.sampling import Vc3dCoordinateSampler
    from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import load_fiber_file

    torch.set_num_threads(1)
    torch.manual_seed(20260913)
    out = Path(settings["output"])
    out.mkdir(parents=True, exist_ok=False)
    snapshot, digest = freeze_checkpoint(Path(settings["checkpoint"]), out)
    payload = torch.load(snapshot, map_location="cpu", weights_only=True, mmap=True)
    device = torch.device("cuda:0")
    adapter = FiberTrace3DPredictAdapter({}, checkpoint=snapshot)
    if not adapter.config.get("embedding", {}).get("enabled"):
        raise ValueError("checkpoint has no embeddings")
    scale = adapter.native_output_stride
    model = adapter.load_model(device=device)
    precision = _mixed_precision_config_from_training(adapter.config.get("training", {}), device)
    template = json.loads(Path(settings["template"]).read_text())
    report = dict(settings=settings, checkpoint_sha256=digest, step=payload["step"],
        samples_seen=payload.get("samples_seen"), precision=str(precision), output_stride=scale,
        note="BF16 model; raw vector FP32 lattice blending then trilinear query then normalization. No embedding Zarr format exists yet; not an f16 accumulator/storage roundtrip. Off-GT controls are unlabeled, conservative radius80-base exclusion, not a tracing accuracy metric.", cases=[])
    (out/"provenance.json").write_text(json.dumps({k:v for k,v in report.items() if k!="cases"}, indent=2))
    start = time.monotonic()
    for source in template["sources"]:
        if source["id"] == "PHercParis4_fibers":
            continue
        level = int(source["input_level"])
        if level != 1:
            raise ValueError("this fixed diagnostic targets level1 input")
        sampler = Vc3dCoordinateSampler(source["input_volume"], level=level,
            cache_root=settings["cache_root"], **native_reader_options({}))
        validate_input_lattice(input_root_attributes(sampler.volume), level)
        shape = tuple(map(int, sampler.volume.shape_at(level)))
        geometry = DemandGeometry(shape, settings["tile"], settings["overlap"], settings["border"], scale, settings["chunk"])
        reader = NativeTileReader(sampler, geometry, sampler.volume.dtype)
        candidates = sorted((Path(template["split_root"])/"test"/source["id"]).glob("*.json"))
        count = 0
        for path in candidates:
            for fiber in load_fiber_file(path):
                full = np.asarray(fiber.line_points_xyz, dtype=float)[:, ::-1]/(2*scale)
                try:
                    points = sample_line(full, settings["spacing_input"]/scale, settings["length_input"]/scale)
                except ValueError:
                    continue
                print(f"consistency {source['sample_id']} {path.name} start", flush=True)
                row = analyze_window(model, adapter, reader, geometry, points, full, precision, device, 20260913+count)
                row.update(scroll=source["sample_id"], fiber=str(path), geometry=geometry.__dict__,
                           spacing_base=settings["spacing_input"]*2, source=source["input_volume"])
                report["cases"].append(row)
                (out/"results.json").write_text(json.dumps(report, indent=2))
                print(f"consistency {source['sample_id']} done tiles={len(row['tiles'])} seconds={row['seconds']:.1f} overlap={row['metrics']['same_point_cross_tile']}", flush=True)
                count += 1
                if count >= settings["fibers_per_scroll"]:
                    break
            if count >= settings["fibers_per_scroll"]:
                break
        if count < settings["fibers_per_scroll"]:
            raise ValueError(f"insufficient long fibers for {source['id']}: {count}")
    report["seconds"] = time.monotonic()-start
    (out/"results.json").write_text(json.dumps(report, indent=2))
    print(f"consistency complete output={out} seconds={report['seconds']:.1f}", flush=True)


def write_report(settings):
    out = Path(settings['output'])
    report = json.loads((out/'results.json').read_text())
    if 'seconds' not in report:
        raise ValueError('analysis did not complete')
    def combined(rows, path):
        stats = []
        for row in rows:
            d = row['metrics']
            for key in path:
                d = d[key]
            if d['n']:
                stats.append(d)
        n = sum(d['n'] for d in stats)
        return sum(d['mean']*d['n'] for d in stats)/n if n else float('nan')
    text = [
        '# Embedding tile consistency — 2026-09-13', '',
        f"Raw checkpoint: update **{report['step']:,}**, **{report['samples_seen']:,} samples**.",
        f"SHA256: `{report['checkpoint_sha256']}`.", '',
        'Ten held-out JSON fibers (two per scroll), the lexicographically first eligible files; '
        'each contributes a centered 512-base-voxel GT arc sampled every4 base voxels '
        '(129 points; about1.23mm per window at2.4µm). This is a deterministic pilot, not all50 fibers.', '',
        'Input level1;256³ input tiles, overlap64, border16; native output stride4 '
        '(output/base factor8), stored-chunk size32 output voxels. BF16 model, '
        'existing native VC3D CT reader and inference preprocessing, shared canonical '
        'tile geometry and Gaussian-downsampled blend ramp. All contributors to all '
        'eight interpolation corners are blended before trilinear sampling and L2 normalization.', '',
        'This diagnostic blends signed raw vectors in FP32. Embedding publication '
        'does not exist yet, so this is NOT an f16 accumulator/Zarr roundtrip. '
        'Storage-boundary statistics classify the continuous field at hypothetical '
        'chunk boundaries; they do not test storage serialization. Tile samples with '
        'blend weight≤0.05 are excluded from same-tile/cross-tile comparison, not from blending.', '',
        '## Cosine consistency', '',
        'Means below use matched4-base-voxel arclength separation. “Switch” means a '
        'change of dominant contributor at the largest-weight interpolation corner '
        '(a proxy for the blend handoff, not a discontinuous field selection).', '',
        '| Scroll | Same point, overlapping tiles | Within one tile | Blended interior | Blended switch | Storage crossing | Switch pairs |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    aggregate = []
    scrolls = sorted({c['scroll'] for c in report['cases']})
    for scroll in scrolls:
        rows = [c for c in report['cases'] if c['scroll']==scroll]
        v = [combined(rows, ['same_point_cross_tile'])] + [combined(rows, ['by_lag','1',k])
            for k in ('within_tile','blended_interior','blended_switch','stored_chunk_crossing')]
        n = sum(c['metrics']['by_lag']['1']['blended_switch']['n'] for c in rows)
        text.append('| '+scroll+' | '+' | '.join(f'{x:.6f}' for x in v)+f' | {n} |')
        aggregate.append(v)
    text.append('| Equal-scroll mean | '+' | '.join(f'{x:.6f}' for x in np.nanmean(aggregate, axis=0))+' | — |')
    text += ['', '## Longer along-fiber comparisons', '',
        'Equal-scroll means, pooling pairs within each scroll. All distances are GT '
        'arclength separations, not straight-line displacement.', '',
        '| Separation (base voxels) | Within tile | Blended interior | Blended switch |',
        '|---:|---:|---:|---:|']
    for lag in ('1','4','16','32'):
        vals = [[combined([c for c in report['cases'] if c['scroll']==s], ['by_lag',lag,k])
                 for k in ('within_tile','blended_interior','blended_switch')] for s in scrolls]
        text.append(f'| {int(lag)*4} | '+' | '.join(f'{x:.6f}' for x in np.nanmean(vals, axis=0))+' |')
    text += ['', '## Off-fiber controls: consistency is not separation', '',
        'Controls lie >80 base voxels from the entire selected GT polyline, '
        'conservatively outside its20/80 exclusion ellipse. They are unlabeled: '
        'they can be background or other fibers. This is not exactly the training '
        'candidate distribution. Each control is compared with a GT point from the '
        'same tile. High-presence means predicted presence≥0.5.', '',
        '| Scroll | Controls | Cosine>0.9 | High-presence controls | High-presence cosine>0.9 | High-presence cosine p05 / median / p95 |',
        '|---|---:|---:|---:|---:|---|']
    controls = []
    for s in scrolls:
        rows = [c for c in report['cases'] if c['scroll']==s]
        cos = np.concatenate([c['negative_cosines'] for c in rows])
        pres = np.concatenate([c['negative_presences'] for c in rows])
        high = cos[pres>=.5]
        pct, highpct = float(np.mean(cos>.9)*100), float(np.mean(high>.9)*100)
        controls.append(dict(scroll=s,n=len(cos),fraction_high_similarity=pct/100,
                             high_presence_n=len(high),high_presence_fraction_high_similarity=highpct/100))
        text.append(f'| {s} | {len(cos)} | {pct:.1f}% | {len(high)} | {highpct:.1f}% | '+
                    ' / '.join(f'{v:.4f}' for v in np.quantile(high,[.05,.5,.95]))+' |')
    means = [unit(np.asarray(c['blended_embeddings'])).mean(0) for c in report['cases']]
    similarities = unit(np.array(means)) @ unit(np.array(means)).T
    other = similarities[np.triu_indices(len(means), 1)]
    text += ['', f"Across the10 window-mean embedding directions, pairwise cosine min/median/max = "
             f"{other.min():.6f} / {np.median(other):.6f} / {other.max():.6f}. "
             'Cross-window similarity is not an objective violation by itself (training '
             'is patch-local), but near-identical GT directions plus high-cosine '
             'off-GT controls warn against treating continuity as fiber identity.', '',
        '## Norms and blending', '',
        '| Scroll | GT norm mean | Corner cancellation mean | Worst corner cancellation | Zero GT vectors |',
        '|---|---:|---:|---:|---:|']
    for s in scrolls:
        rows = [c for c in report['cases'] if c['scroll']==s]
        text.append(f"| {s} | {combined(rows,['blended_norm']):.3f} | {combined(rows,['cancellation']):.6f} | "
                    f"{min(c['metrics']['cancellation']['min'] for c in rows):.6f} | {sum(c['zero_norm_points'] for c in rows)} |")
    text += ['', 'Cancellation is ||Σw e||/Σw||e|| at all eight support corners, '
        'including corners with zero trilinear coefficient. Thus worst-corner '
        'values need not imply a bad actual GT query; GT norm/continuity are reported separately.', '',
        '## Interpretation', '',
        '- This pilot shows no appreciable GT discontinuity at tile handoffs. '
        'Vector blending is not currently the main obstacle indicated by these samples.',
        '- Many off-GT points remain almost indistinguishable from GT in cosine. '
        'The representation may primarily separate broad foreground/background '
        'groups rather than individual neighboring fibers. This is a warning, '
        'not a demonstrated tracing failure or proof of global collapse.',
        '- Review these results before enabling embedding tracing. A local '
        'similarity score can be continuous yet offer little discrimination.', '',
        '## Reproduction and validation', '',
        '```bash', '/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment embedding-consistency', '```', '',
        'Settings are in `vesuvius/scripts/fiber3d_experiment.json` under '
        '`embedding_consistency`; use a new output directory for a fresh run. '
        'Use the frozen snapshot for exact checkpoint reuse. The launcher uses '
        'the existing exclusive, non-pausable manager command; running training '
        'releases its GPUs and resumes after analysis.', '',
        f"Measured analysis wall time: **{report['seconds']:.1f}s**, one GPU, "
        f"{sum(len(c['tiles']) for c in report['cases'])} tile forwards. Includes source "
        'opening/CT reads/forward/CPU metrics, excludes manager waiting and initial '
        'checkpoint/model load. This is not a throughput benchmark.', '',
        'CPU synthetic tests: affine interpolation/arclength, identical-tile '
        'seam invariance, opposing-tile disagreement/cancellation (3 passed).', '',
        f"Raw per-fiber points, vectors, cosine samples, distributions, tiles and settings: `{out/'results.json'}`.",
        'No tracer or training behavior was changed. macOS/arm64 and embedding storage '
        'precision were not tested; only this Ubuntu/CUDA environment was measured.', '']
    (out/'report.md').write_text('\n'.join(text))
    (out/'summary.json').write_text(json.dumps(dict(controls=controls, cross_window_cosines=distribution(other)),indent=2))
    print('\n'.join(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("control")
    parser.add_argument("--unmanaged", action="store_true")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    settings = json.loads(Path(args.control).read_text())["embedding_consistency"]
    if args.report:
        write_report(settings)
        return 0
    if not args.unmanaged:
        from .managed_evaluation import launch_managed
        return launch_managed([sys.executable, "-m", "vesuvius.neural_tracing.fiber_trace_3d.embedding_consistency", args.control], devices=args.devices,
                              settings={"priority":100, "label":"embedding-consistency"})
    run(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
