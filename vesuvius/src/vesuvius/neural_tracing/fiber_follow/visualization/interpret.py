"""Generate a seven-page measured interpretation atlas for the patch4 slab follower."""
import argparse
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shlex
import sys
from unittest.mock import patch

import numpy as np
import torch

from ..regression.data import ObservationBuilder
from ..regression.history_slabs import slab_layout
from ..regression.model import TOKEN_ARCHITECTURE
from ..regression.train import load_checkpoint
from ..shared.data import (SampleConfig, fiber_manifest, label_state, load_fibers,
                           OnPolicyStates, split_fibers, ZBand)
from ..shared.geometry import arclength, frame_from_heading, interp_at, tangent_at
from ..shared.policy import commit_prefix
from ..shared.reference import SEED_FIELDS, observed_path, observed_seed
from ..shared.volume import FiberVolume
from .capture import array, capture, validate_attention


VARIANTS = {
    'baseline': 'Full historical slabs',
    'no_generator_history': 'Generator history read disabled',
    'no_scorer_history': 'Scorer history read disabled',
    'no_history': 'Both history reads disabled',
    'seed_only': 'Seed slab only',
}


def legacy_prefix(hist, mask, pos, seed, seed_age, history_step):
    """v5 has sampled history, not the complete committed polyline required by v6.

    Only adapt histories reaching the seed: never bridge an unobserved long gap.
    The short seed-to-oldest-sample segment is an explicit resampling approximation.
    """
    mask = np.asarray(mask).astype(bool)
    count = int(mask.sum())
    if not np.array_equal(mask, np.arange(len(mask)) < count):
        raise ValueError('Legacy history must be a contiguous observed prefix')
    history = np.asarray(hist)[mask][::-1]
    first = history[0] if len(history) else pos
    if seed_age > (count + 1.5) * history_step or np.linalg.norm(first - seed) > 1.5 * history_step:
        raise ValueError('Legacy replay truncates the seed path; use replay v6 with complete prefixes')
    path = np.concatenate((np.asarray(seed)[None], history, np.asarray(pos)[None]))
    return path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9]]


def replay_item(source, row, fibers, sample):
    """Read current replay through its loader, or explicitly adapt read-only v5 arrays."""
    source = Path(source)
    if source.is_dir():
        metadata = json.loads((source / 'metadata.json').read_text())
    else:
        with np.load(source) as archive:
            metadata = json.loads(str(archive['__metadata__'].item()))
    version = metadata['version']
    if version == 6:
        states = OnPolicyStates.load(source)
        states.validate_fibers(fibers)
        if not 0 <= row < len(states):
            raise ValueError('Replay row outside cache')
        get = lambda key: getattr(states, key)[row]
        prefix = states.observed_prefix(row)
        provenance = 'Complete committed replay-v6 prefix'
    elif version == 5 and source.is_dir():
        if metadata['fibers'] != fiber_manifest(fibers):
            raise ValueError('Replay fibers do not match the selected fiber split')
        def get(key):
            values = np.load(source / (key + '.npy'), mmap_mode='r')
            if not 0 <= row < len(values):
                raise ValueError('Replay row outside cache')
            return np.array(values[row], copy=True)
        if not bool(get('seed_valid')):
            raise ValueError('Legacy adaptation requires a saved seed')
        prefix = legacy_prefix(get('hist'), get('hmask'), get('pos'), get('seed_pos'),
                               float(get('seed_age')), sample.history_step)
        provenance = ('Legacy v5: saved resampled observed history plus saved seed; '
                      'not the original per-commit polyline. No annotation-derived history.')
    else:
        raise ValueError('Use a replay-v6 cache or an existing v5 mmap directory')
    index = int(get('fiber_idx'))
    item = label_state(fibers[index], get('pos'), get('frame'), get('hist'), get('hmask'), sample,
                      t=float(get('t')), reverse=bool(get('reverse')), offtrack=bool(get('offtrack')))
    item.update({key: get(key) for key in SEED_FIELDS})
    item['observed_path'] = prefix
    observed_path(item)  # Validate endpoints before reading any volume.
    return item, index, provenance


def annotation_item(fiber, at, reverse, sample):
    """Deterministic unperturbed annotated state for a new fiber, without replay."""
    if not 0 <= at <= fiber.length:
        raise ValueError(f'--arc-position must be between 0 and {fiber.length:g} trace voxels')
    points = fiber.points[::-1] if reverse else fiber.points
    arc = arclength(points)
    pos = interp_at(points, arc, np.array([at]))[0]
    frame = frame_from_heading(tangent_at(points, arc, at))
    history_at = at - np.arange(1, sample.n_history + 1) * sample.history_step
    hist = interp_at(points, arc, history_at.clip(0, fiber.length))
    mask = (history_at >= 0).astype(np.float32)
    item = label_state(fiber, pos, frame, hist, mask, sample,
                       t=fiber.length-at if reverse else at, reverse=reverse)
    # Entire known annotation prefix is explicitly a synthetic observed path.
    item['observed_path'] = np.concatenate((points[arc < at], pos[None]))
    seed_local = (points[:1] - pos) @ frame
    item.update(observed_seed(pos, frame, seed_local, np.ones(1)))
    item['seed_tangent'] = tangent_at(points, arc, 0.)
    item['seed_age'] = at
    return item


def analyze(model, x, hist, hmask, captured, threshold, n_commit):
    saved, metrics, stats = {}, {}, []
    fixed = torch.from_numpy(captured['points'])[None, None]
    names = list(VARIANTS) + [f'without_block_{i+1}' for i in range(len(model.encoder.blocks))]
    with torch.inference_mode():
        for name in names:
            inputs = dict(x)
            with ExitStack() as stack:
                if name in ('no_generator_history', 'no_history'):
                    stack.enter_context(patch.object(model.history_attention, 'forward_cached', lambda query, *args: query))
                if name in ('no_scorer_history', 'no_history'):
                    stack.enter_context(patch.object(model.confidence_scorer.history_attention, 'forward_cached', lambda query, *args: query))
                if name == 'seed_only':
                    inputs['history_valid'] = x['history_valid'].clone()
                    inputs['history_valid'][:, 1:] = False
                if name.startswith('without_block_'):
                    block = model.encoder.blocks[int(name.rsplit('_', 1)[1]) - 1]
                    stack.enter_context(patch.object(block, 'forward', lambda value: value))
                out = model(inputs, hist, hmask, candidates=fixed,
                            confidence_threshold=threshold, n_commit=n_commit)
            count, allowed = commit_prefix(out['points'], out['confidence'], threshold, n_commit,
                                           model.cfg.max_recovery_distance)
            shift = (out['points'] - fixed[:, 0]).norm(dim=-1)[0]
            metrics[name] = dict(commit=int(count[0]), connection_allowed=bool(allowed[0]),
                                 selected_attempt=int(out['selected_refinement'][0]),
                                 attempts=out['refinement_points'].shape[1],
                                 mean_path_shift=float(shift.mean()), max_path_shift=float(shift.max()),
                                 last_confidence=float(out['confidence'][0, -1]),
                                 fixed_curve_last_confidence=float(out['candidate_confidence'][0, 0, -1]))
            for key, value in out.items():
                saved[name + '_' + key] = array(value[0])
            if name == 'baseline':
                for key in ('points', 'confidence', 'refinement_points', 'refinement_confidence'):
                    np.testing.assert_array_equal(saved[name + '_' + key], captured[key])
                np.testing.assert_allclose(out['candidate_confidence'][0, 0], out['confidence'][0], rtol=1e-5, atol=1e-6)
            print(name, metrics[name], flush=True)
        restored = model(x, hist, hmask, confidence_threshold=threshold, n_commit=n_commit)
        np.testing.assert_array_equal(array(restored['points'][0]), captured['points'])
        np.testing.assert_array_equal(array(restored['confidence'][0]), captured['confidence'])
    for i in range(len(model.encoder.blocks)):
        before = captured['encoder_input' if i == 0 else f'axial_{i-1}']
        after = captured[f'axial_{i}']
        norm = np.linalg.norm(before, axis=-1)
        relative = np.linalg.norm(after - before, axis=-1) / norm.clip(1e-12)
        cosine = (before * after).sum(-1) / (norm * np.linalg.norm(after, axis=-1)).clip(1e-12)
        angle = np.degrees(np.arccos(cosine.clip(-1, 1)))
        saved[f'block_{i+1}_relative_change'] = relative
        stats.append(dict(block=i+1, median_relative_change=float(np.median(relative)),
                          median_angle_degrees=float(np.median(angle))))
    return saved, dict(metrics=metrics, encoder_blocks=stats, interventions=VARIANTS,
                      protocol='One decision, fixed inputs, EMA FP32. Fixed-curve scores use the baseline selected curve. '
                               'History-read interventions retain current-crop references and features. '
                               'Attention is read allocation, not causal importance; interventions are not accuracy estimates.')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Saved current patch4 token/slab checkpoint (EMA)')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ct', help='CT zarr root; default: checkpoint')
    parser.add_argument('--fiber-zarrs', help='Presence/nx/ny zarr directory; default: checkpoint')
    for name, kind in [('ct-level', int), ('fiber-level', int), ('ct-grid-scale', float), ('grid-scale', float)]:
        parser.add_argument('--' + name, type=kind, help='Volume grid override; default: checkpoint')
    parser.add_argument('--fibers', help='Fiber JSON directory; default: checkpoint training options')
    parser.add_argument('--fiber-split', choices=['train', 'val', 'all'], default='train')
    parser.add_argument('--val-z', type=float, nargs=2, help='Split band in base voxels; default: checkpoint')
    parser.add_argument('--replay', type=Path, help='Replay cache containing the decision')
    parser.add_argument('--row', type=int, default=0)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument('--fiber-index', type=int, help='Index within the selected fiber split')
    selector.add_argument('--fiber-name', help='Exact fiber JSON basename')
    parser.add_argument('--arc-position', type=float, help='Without replay: distance from traversal start in trace voxels')
    parser.add_argument('--reverse', action='store_true', help='Reverse annotation traversal without replay')
    parser.add_argument('--compare-atlas', type=Path, help='Validate decision identity and compare inputs with an older atlas')
    parser.add_argument('--confidence-threshold', type=float, default=.5)
    parser.add_argument('--n-commit', type=int, help='Default: checkpoint')
    parser.add_argument('--threads', type=int, default=4)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.threads < 1 or not 0 <= args.confidence_threshold <= 1:
        raise ValueError('Positive thread count and confidence threshold in [0,1] required')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    cp = args.checkpoint.resolve()
    digest = hashlib.sha256(cp.read_bytes()).hexdigest()
    model, _, _, spec, ck = load_checkpoint(cp, 'cpu')
    if model.architecture != TOKEN_ARCHITECTURE:
        raise ValueError(f'Only {TOKEN_ARCHITECTURE} is supported')
    overrides = {key: getattr(args, arg) for key, arg in [('ct_zarr', 'ct'), ('fiber_zarr_dir', 'fiber_zarrs'),
                  ('ct_level', 'ct_level'), ('fiber_level', 'fiber_level'), ('ct_grid_scale', 'ct_grid_scale'),
                  ('grid_scale', 'grid_scale')] if getattr(args, arg) is not None}
    spec = replace(spec, **overrides)
    cfg = model.cfg
    n_commit = ck['n_commit'] if args.n_commit is None else args.n_commit
    if not 1 <= n_commit <= cfg.n_future:
        raise ValueError('--n-commit must fit the model horizon')
    fibers_path = args.fibers or ck['training_options']['fibers']
    fibers = load_fibers(fibers_path, grid_scale=spec.grid_scale)
    val_z = args.val_z or ck['training_options']['val_z']
    if args.fiber_split != 'all':
        train, val = split_fibers(fibers, ZBand(*(z/spec.grid_scale for z in val_z)))
        fibers = train if args.fiber_split == 'train' else val
    sample = SampleConfig(**dict(ck['sample_cfg'], crop=cfg.fine))
    index = args.fiber_index
    if args.fiber_name is not None:
        matches = [i for i, f in enumerate(fibers) if f.name == args.fiber_name]
        if len(matches) != 1:
            raise ValueError('Fiber name must match exactly one fiber in the selected split')
        index = matches[0]
    if args.replay:
        if args.arc_position is not None or args.reverse:
            raise ValueError('Replay supplies arc position and direction')
        item, replay_index, history_source = replay_item(args.replay, args.row, fibers, sample)
        if index is not None and index != replay_index:
            raise ValueError('Selected fiber does not match replay row')
        index = replay_index
    else:
        if index is None or args.arc_position is None:
            raise ValueError('Supply --replay, or a fiber selector and --arc-position')
        if not 0 <= index < len(fibers):
            raise ValueError('Fiber index outside selected split')
        item = annotation_item(fibers[index], args.arc_position, args.reverse, sample)
        history_source = 'Synthetic clean annotation prefix (no perturbation)'
    fiber = fibers[index]
    dest = args.output_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.glob('*.npz')):
        raise ValueError('Output already contains an atlas; choose a fresh output directory')
    print(f'Loading volume inputs for {fiber.name}', flush=True)
    x = ObservationBuilder(cfg).images([item], FiberVolume(spec, cache_bytes=256 << 20))
    hist = torch.as_tensor(np.array(item['hist_local'])[None], dtype=torch.float32)
    hmask = torch.as_tensor(np.array(item['hmask'])[None], dtype=torch.float32)
    with torch.inference_mode():
        with capture(model) as arrays:
            out = model(x, hist, hmask, confidence_threshold=args.confidence_threshold, n_commit=n_commit)
        check = model(x, hist, hmask, confidence_threshold=args.confidence_threshold, n_commit=n_commit)
        for key in out:
            torch.testing.assert_close(out[key], check[key], rtol=0, atol=0)
            arrays[key] = array(out[key][0])
        arrays.update(input=array(x['fine'][0]), hist=array(hist[0]), hmask=array(hmask[0]),
                      token_xyz=array(model.encoder.token_xyz).reshape(*cfg.token_shape, 3),
                      segment_samples=array(model.confidence_scorer.segment_samples(out['points'])[0]),
                      plane_ab=item['plane_ab'], plane_mask=item['plane_mask'], observed_path=observed_path(item))
        for key, value in x.items():
            if key != 'fine' and key != 'history_load_seconds':
                arrays['input_' + key] = array(value[0])
    layout = slab_layout(item)
    arrays['slab_positions'] = np.stack([s['pos'] for s in layout])
    arrays['slab_frames'] = np.stack([s['frame'] for s in layout])
    validate_attention(arrays)
    comparison = None
    if args.compare_atlas:
        prior = json.loads((args.compare_atlas / 'sample_provenance.json').read_text())
        with np.load(args.compare_atlas / 'sample_activations.npz') as old:
            assert prior['fiber_source_hash'] == fiber.source_hash
            np.testing.assert_array_equal(prior['position_xyz_trace_grid'], item['pos'])
            np.testing.assert_array_equal(prior['frame'], item['frame'])
            np.testing.assert_array_equal(old['hist'], arrays['hist'])
            np.testing.assert_array_equal(old['hmask'], arrays['hmask'])
            comparison = dict(atlas=str(args.compare_atlas.resolve()), same_decision=True,
                              input_max_abs_difference_by_channel=np.abs(old['input']-arrays['input']).reshape(cfg.input_channels, -1).max(1).tolist())
    saved, report = analyze(model, x, hist, hmask, arrays, args.confidence_threshold, n_commit)
    for collection in (arrays, saved):
        for key, value in collection.items():
            assert np.isfinite(value).all(), key
    assert hashlib.sha256(cp.read_bytes()).hexdigest() == digest, 'Checkpoint changed during run; use a numbered checkpoint'
    meta = dict(checkpoint=str(cp), checkpoint_sha256=digest, checkpoint_step=ck['step'], weights='EMA',
                model_cfg=cfg.to_dict(), architecture=model.architecture, volume_spec=spec.to_dict(),
                fibers=str(Path(fibers_path).resolve()), fiber_split=args.fiber_split, val_z=val_z,
                source=str(args.replay.resolve()) if args.replay else 'annotation',
                replay_row=args.row if args.replay else None, fiber_index=index, fiber_name=fiber.name,
                fiber_source_hash=fiber.source_hash, position_xyz_trace_grid=item['pos'].tolist(),
                frame=item['frame'].tolist(), history_source=history_source, observations=len(layout),
                history_points=int(hmask.sum()), n_commit=n_commit, confidence_threshold=args.confidence_threshold,
                selected_refinement=int(arrays['selected_refinement']), attempts=len(arrays['refinement_points']),
                augmentation=False, precision='CPU FP32', threads=args.threads, seed=0,
                attention_diagnostic_precision='FP32 Q/K logits; FP64 softmax and head mean; FP32 stored weights',
                parameter_count=sum(p.numel() for p in model.parameters()), prior_comparison=comparison,
                command=shlex.join([sys.executable, '-m', 'vesuvius.neural_tracing.fiber_follow.visualization.interpret'] + (sys.argv[1:] if argv is None else argv)),
                shapes={key: list(value.shape) for key, value in arrays.items()})
    np.savez_compressed(dest / 'sample_activations.npz', **arrays)
    np.savez_compressed(dest / 'analysis_arrays.npz', **saved)
    (dest / 'sample_provenance.json').write_text(json.dumps(meta, indent=2))
    (dest / 'analysis_summary.json').write_text(json.dumps(report, indent=2))
    from .render import render
    from .validate import validate
    render(dest)
    validate(dest)
    print(f'Atlas: {dest / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
