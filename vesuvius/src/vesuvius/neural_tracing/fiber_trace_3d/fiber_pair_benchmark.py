"""Freeze and geometrically audit an auxiliary fiber for a single-case benchmark."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import numpy as np

from lasagna.inference_provenance import atomic_write, sha256_file
from .embedding import nearest_polyline_points


def distance_profile(points, target):
    """Exact sampled-point to polyline distances, not a topology test."""
    points = np.asarray(points, dtype=np.float64)
    nearest, squared = nearest_polyline_points(points, target)
    arcs = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return dict(arc_base=arcs.tolist(), points_base_xyz=points.tolist(),
                distance_base=np.sqrt(squared).tolist(), nearest_base_xyz=nearest.tolist())


def build_reference_plan(evaluation, fiber_paths, binary, config_path, plan_path,
                         expected_sha256=None):
    """Build one native CP-bounded reference plan for an explicit fiber list."""
    fiber_paths = [Path(path) for path in fiber_paths]
    if expected_sha256 is not None:
        if len(expected_sha256) != len(fiber_paths):
            raise ValueError("fiber path/hash counts differ")
        for path, expected in zip(fiber_paths, expected_sha256):
            if sha256_file(path) != expected:
                raise ValueError(f"fiber annotation changed: {path}")
    config = deepcopy(evaluation)
    config['sources'][0]['fibers'] = [str(path) for path in fiber_paths]
    config.pop('reference_plan', None)
    atomic_write(config_path, config)
    plan = json.loads(subprocess.check_output(
        [str(binary), 'replay-references', '--config', str(config_path)], text=True))
    atomic_write(plan_path, plan)
    return plan


def write_pair_benchmark(root, evaluation, manifest, settings, binary):
    root = Path(root)
    case, = manifest['cases']
    source = Path(settings['path'])
    if sha256_file(source) != settings['sha256']:
        raise ValueError('auxiliary annotation changed')
    directory = root / 'pair'
    directory.mkdir()
    auxiliary = directory / source.name
    auxiliary.write_bytes(source.read_bytes())
    if sha256_file(auxiliary) != settings['sha256']:
        raise ValueError('auxiliary annotation changed while copying')
    primary = directory / Path(case['path']).name
    primary.write_bytes(Path(case['path']).read_bytes())
    if sha256_file(primary) != case['sha256']:
        raise ValueError('primary annotation changed')
    config_path = directory / 'reference_config.json'
    plan = build_reference_plan(evaluation, [primary, auxiliary], binary, config_path,
        directory / 'reference_plan.json', [case['sha256'], settings['sha256']])
    a, b = plan['sources'][0]['fibers']
    if len(a['runs']) != 1 or len(b['runs']) != 1:
        raise ValueError('pair pilot requires one eligible run per fiber')
    reference = a['runs'][0]['points_base_xyz']
    neighbor = b['runs'][0]['points_base_xyz']
    trace_path, = root.glob('scrolls/*/cases/*/trace.json')
    replay = json.loads(trace_path.read_text())['replay']
    traced = replay['segments'][0]['trace_points_base_xyz']
    profiles = dict(reference_to_auxiliary=distance_profile(reference, neighbor),
                    auxiliary_to_reference=distance_profile(neighbor, reference),
                    trace_to_auxiliary=distance_profile(traced, neighbor),
                    auxiliary_to_trace=distance_profile(neighbor, traced),
                    trace_to_reference=distance_profile(traced, reference))
    atomic_write(directory / 'geometry_profiles.json', profiles)
    info = dict(primary=dict(path=str(primary), sha256=case['sha256'], role='primary reference'),
                auxiliary=dict(path=str(auxiliary), sha256=settings['sha256'], original=str(source),
                               role='user-supplied neighboring-fiber hypothesis; not independent GT'),
                trace=str(trace_path), evaluation=str(root / 'evaluation.json'),
                checkpoint_sha256=settings['checkpoint_sha256'],
                coordinate_space=evaluation['sources'][0]['coordinate_space'],
                limitations=['Euclidean distances in base voxels, not acceptance-ellipse errors.',
                             'Native CP-bounded lines only; no extrapolated tails.',
                             'First trace segment only; no bridge across restart.',
                             'Distances at stored vertices, no unweighted quantile claims.',
                             'Geometry does not establish physical identity or relative prediction strength.'])
    atomic_write(directory / 'manifest.json', info)
    lines = ['# Single-case touching-fiber benchmark', '',
             'Primary reference and auxiliary neighbor hypothesis are frozen here; see manifest.json.',
             'Baseline rendering and metrics: ../README.md and ../metrics.json.', '',
             '| Profile | Length base vx | Start distance | Smallest sampled distance | End distance |',
             '|---|---:|---:|---:|---:|']
    for name, profile in profiles.items():
        d = profile['distance_base']
        lines.append(f'| {name} | {profile["arc_base"][-1]:.3f} | {d[0]:.3f} | {min(d):.3f} | {d[-1]:.3f} |')
    lines += ['', *info['limitations'], '',
              'Future target: explain both paths jointly, allow shared evidence in the merged region,',
              'and preserve branch identity after separation. No global-assignment solver implemented.']
    (directory / 'README.md').write_text('\n'.join(lines)+'\n')
    return info
