#!/usr/bin/env python3
"""Find graph patches that overlap measured sources of existing fiber strips.

The existing strip includes generated fiber geometry. A candidate counts only
where the *same* candidate vertices hit both an original assigned patch and
the exported strip. This excludes fiber-only contacts and source patches that
were dropped from the exported component. Discovery is read-only; see --help.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.ndimage import label

from connect_overlapping_patches import build_surface_index, patch_query_points
from tifxyz import load_tifxyz
from vc_spiral import surface_index


DEFAULT_OUTPUTS = Path(__file__).resolve().parent / 'outputs'
DEFAULT_PATCH_ROOT = Path('/home/sean/Desktop/graph_patch_merging')


def patch_catalog(root):
    result = {}
    for path in sorted(root.iterdir()):
        metadata = path / 'meta.json'
        if not path.is_dir() or not metadata.is_file():
            continue
        value = json.loads(metadata.read_text())
        bbox = np.asarray(value['bbox'], dtype=np.float64)
        if bbox.shape != (2, 3) or not np.isfinite(bbox).all():
            raise ValueError(f'Invalid bbox: {metadata}')
        result[path.name] = (path, bbox)
    return result


def strip_sources(outputs, strip_row):
    batch = strip_row['source'].split('/')[0]
    stem = Path(strip_row['name']).stem
    status_path = outputs / batch / 'jobs' / stem / 'status.json'
    status = json.loads(status_path.read_text())
    report = json.loads((outputs / batch / status['report']).read_text())
    if status['status'] != 'exported':
        raise ValueError(f'{stem}: listed strip is not exported')
    return (outputs / batch / status['report']).parent / 'review.json', [
        item.removeprefix('patch:') for item in report.get('assigned', [])
    ]


def build_shared_index(available, rows, outputs, tolerance):
    """Load graph patches and the selected strips into one reusable surface index."""
    surfaces = []
    positions = {}
    empty = []

    def add(key, path):
        patch = load_tifxyz(path, skip_empty=True)
        if patch is None:
            empty.append(str(path))
            return
        zyx = patch.zyxs.numpy()
        scale = patch.scale.numpy()
        positions[key] = len(surfaces)
        surfaces.append(surface_index.QuadSurface(f'{key[0]}:{key[1]}', zyx,
                                                  float(scale[0]), float(scale[1])))

    for number, (patch_id, (path, _bbox)) in enumerate(available.items(), 1):
        add(('graph', patch_id), path)
        if number % 5000 == 0:
            print(f'Loaded {number}/{len(available)} graph patches for shared index', flush=True)
    for row in rows:
        name = row['name']
        add(('strip', name), outputs / 'fiber_merge_all_strips' / 'strips_flattened' / name)
    print(f'Building one surface index over {len(surfaces)} graph patches and strips '
          f'({len(empty)} empty surfaces skipped)', flush=True)
    index = surface_index.SurfacePatchIndex()
    index.rebuild(surfaces, bbox_padding=tolerance, sampling_stride=1)
    return index, positions, empty


def hits(index, xyz, tolerance, subset=None):
    if subset is None:
        offsets, _surface, _distance, _ij = index.locate_all_xyz_batch(xyz, tolerance)
    else:
        offsets, _surface, _distance, _ij = index.locate_all_xyz_batch_in(xyz, subset, tolerance)
    return np.diff(offsets) > 0


def overlap_counts(index, candidate_surface, source_subset, strip_subset, tolerance, stride):
    xyz = index.sample_xyz(candidate_surface, stride)
    if not len(xyz):
        return {'samples': 0, 'source_hits': 0, 'strip_hits': 0, 'joint_hits': 0}
    subset = np.concatenate((source_subset, strip_subset))
    offsets, surfaces, _distance, _ij = index.locate_all_xyz_batch_in(xyz, subset, tolerance)
    point = np.repeat(np.arange(len(xyz)), np.diff(offsets))
    strip_match = surfaces == strip_subset[0]
    source = np.zeros(len(xyz), dtype=bool)
    strip = np.zeros(len(xyz), dtype=bool)
    source[point[~strip_match]] = True
    strip[point[strip_match]] = True
    return {'samples': len(xyz), 'source_hits': int(source.sum()),
            'strip_hits': int(strip.sum()), 'joint_hits': int((source & strip).sum())}


def scan_strip(strip_row, outputs, candidates, index, positions,
               candidate_surfaces, nearby_cache, args):
    name = strip_row['name']
    review_path, source_ids = strip_sources(outputs, strip_row)
    result = {'strip': name, 'source_count': len(source_ids), 'candidates': [],
              'nearby_candidates': 0, 'reason': None, 'review': str(review_path)}
    if not source_ids:
        result['reason'] = 'fiber_only'
        return result
    missing = [source_id for source_id in source_ids if ('graph', source_id) not in positions]
    if missing:
        raise ValueError(f'{name}: {len(missing)} measured source patches are missing')
    nearby = set()
    for source_id in source_ids:
        if source_id not in nearby_cache:
            nearby_cache[source_id] = set(index.nearby_surfaces(positions['graph', source_id]))
        nearby.update(nearby_cache[source_id])
    strip_surface = positions['strip', name]
    nearby.intersection_update(index.nearby_surfaces(strip_surface))
    source_set = set(source_ids)
    candidate_names = sorted(candidate_surfaces[surface] for surface in nearby
                             if surface in candidate_surfaces
                             and candidate_surfaces[surface] not in source_set)
    result['nearby_candidates'] = len(candidate_names)
    if not candidate_names:
        return result
    source_subset = np.asarray([positions['graph', source_id] for source_id in source_ids],
                               dtype=np.int32)
    strip_subset = np.asarray([positions['strip', name]], dtype=np.int32)
    for candidate_id in candidate_names:
        candidate_surface = positions.get(('graph', candidate_id))
        if candidate_surface is None:
            continue
        preliminary = overlap_counts(index, candidate_surface, source_subset, strip_subset,
                                     args.tolerance, args.query_stride)
        if preliminary['joint_hits'] < 8 or preliminary['joint_hits'] / preliminary['samples'] < 0.1:
            continue
        exact = overlap_counts(index, candidate_surface, source_subset, strip_subset,
                               args.tolerance, 1)
        if exact['joint_hits'] < args.min_overlap_points:
            continue
        fraction = exact['joint_hits'] / exact['samples']
        novel = exact['samples'] - exact['strip_hits']
        if fraction < args.min_overlap_fraction or novel < args.min_new_points:
            continue
        result['candidates'].append({'patch': candidate_id, 'path': str(candidates[candidate_id][0]),
                                     'overlap_fraction': fraction, 'overlap_points': exact['joint_hits'],
                                     'new_points': novel, 'total_points': exact['samples']})
    result['candidates'].sort(key=lambda row: (-row['overlap_points'], row['patch']))
    return result


def surface_area_vx2(patch):
    xyz = patch.zyxs.numpy()[..., ::-1].astype(np.float64)
    valid = patch.valid_quad_mask.numpy()
    a, b, c, d = xyz[:-1, :-1], xyz[:-1, 1:], xyz[1:, :-1], xyz[1:, 1:]
    area = (np.linalg.norm(np.cross(b - a, c - a), axis=-1)
            + np.linalg.norm(np.cross(c - d, b - d), axis=-1)) * 0.5
    return float(area[valid].sum())


def verify_merge(original_path, merged_path, accepted_paths, tolerance):
    original = load_tifxyz(original_path)
    merged = load_tifxyz(merged_path)
    _regions, n_regions = label(merged.valid_quad_mask.numpy())
    if n_regions != 1:
        raise ValueError(f'Merged surface has {n_regions} quad regions')
    merged_index = build_surface_index({'merged': merged}, tolerance, 1)
    original_index = build_surface_index({'original': original}, tolerance, 1)
    original_xyz = patch_query_points(original, 1)
    offsets, _surface, distances, _ij = merged_index.locate_all_xyz_batch(original_xyz, tolerance)
    retained = np.diff(offsets) > 0
    retained_fraction = float(retained.mean())
    p95_distance = float(np.percentile(distances, 95)) if len(distances) else float('inf')
    if retained_fraction < 0.999 or p95_distance > 1.0:
        raise ValueError(f'Original strip changed: retained={retained_fraction:.6f}, p95={p95_distance:.3f}')
    additions = []
    for path in accepted_paths:
        candidate = load_tifxyz(path)
        xyz = patch_query_points(candidate, 1)
        novel = ~hits(original_index, xyz, tolerance)
        represented = hits(merged_index, xyz, tolerance)
        new_count = int(novel.sum())
        present_count = int((novel & represented).sum())
        fraction = present_count / max(new_count, 1)
        if present_count < 24 or fraction < 0.4:
            raise ValueError(f'{path.name}: only {present_count}/{new_count} new vertices survived')
        additions.append({'patch': path.name, 'new_vertices': new_count,
                          'represented_new_vertices': present_count, 'represented_fraction': fraction})
    return {'original_retained_fraction': retained_fraction, 'original_distance_p95': p95_distance,
            'original_valid_quads': int(original.valid_quad_mask.sum()),
            'merged_valid_quads': int(merged.valid_quad_mask.sum()), 'additions': additions,
            'area_vx2': surface_area_vx2(merged)}


def write_merged_metadata(original_path, merged_path, native_report, validation):
    before = json.loads((original_path / 'meta.json').read_text())
    rendered = json.loads((merged_path / 'meta.json').read_text())
    result = dict(before)
    result['bbox'] = rendered['bbox']
    result['constraint_graph'] = rendered['constraint_graph']
    result['area_vx2'] = validation['area_vx2']
    if before.get('area_vx2') and before.get('area_cm2'):
        result['area_cm2'] = validation['area_vx2'] * before['area_cm2'] / before['area_vx2']
    result['source'] = 'anchored_strip_merge'
    result['anchored_strip_merge'] = {
        'accepted_patches': [row['patch'] for row in native_report['accepted']],
        'original_retained_fraction': validation['original_retained_fraction'],
        'original_distance_p95': validation['original_distance_p95'],
    }
    (merged_path / 'meta.json').write_text(json.dumps(result, indent=2) + '\n')


def merge_strip(entry, outputs, merge_output, executable, tolerance):
    source = outputs / 'fiber_merge_all_strips' / 'strips_flattened' / entry['strip']
    destination = merge_output / 'strips' / entry['strip']
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f'Output already exists: {destination}')
    candidates = entry['candidates']
    if not candidates:
        shutil.copytree(source, destination)
        return {'status': 'unchanged', 'accepted': [], 'rejected': []}
    (merge_output / 'scratch').mkdir(parents=True, exist_ok=True)
    output_spacing = json.loads((Path(entry['review']).parent / 'report.json').read_text())['output_spacing']
    with tempfile.TemporaryDirectory(prefix='strip-', dir=merge_output / 'scratch') as tmp:
        environment = dict(os.environ, AGENTS_AGENT_MODE='1', OMP_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

        def attempt(rows, name):
            temporary = Path(tmp) / name / entry['strip']
            temporary.parent.mkdir()
            command = [str(executable), entry['review'], str(temporary), str(output_spacing)]
            command += [row['path'] for row in rows]
            process = subprocess.run(command, capture_output=True, text=True, env=environment)
            if process.returncode:
                raise ValueError(f'Merger exit {process.returncode}: {process.stderr.strip()}')
            native = json.loads(process.stdout)
            accepted = {row['patch'] for row in native['accepted']}
            if not accepted:
                return temporary, native, None
            accepted_paths = [Path(row['path']) for row in rows if row['patch'] in accepted]
            validation = verify_merge(source, temporary, accepted_paths, tolerance)
            return temporary, native, validation

        try:
            temporary, native, validation = attempt(candidates, 'all')
            if not native['accepted']:
                shutil.copytree(source, destination)
                return {'status': 'unchanged', **native}
            write_merged_metadata(source, temporary, native, validation)
            temporary.rename(destination)
            return {'status': 'merged', **native, 'validation': validation}
        except Exception as error:
            initial_failure = str(error)

        safe = []
        salvage_rejections = []
        best = None
        attempt_number = 0

        def admit(chunk):
            nonlocal best, attempt_number
            if not chunk:
                return
            attempt_number += 1
            try:
                trial = safe + chunk
                temporary, native, validation = attempt(trial, f'trial_{attempt_number:04d}')
                accepted = {row['patch'] for row in native['accepted']}
                if accepted != {row['patch'] for row in trial}:
                    rejected = next((row for row in native['rejected']
                                     if row['patch'] in {part['patch'] for part in chunk}), None)
                    raise ValueError(rejected['reason'] if rejected else
                                     'Native merger rejected one or more candidates')
                safe.extend(chunk)
                best = temporary, native, validation
            except Exception as error:
                if len(chunk) == 1:
                    salvage_rejections.append({'patch': chunk[0]['patch'], 'reason': str(error)})
                else:
                    midpoint = len(chunk) // 2
                    admit(chunk[:midpoint])
                    admit(chunk[midpoint:])
            print(f"  salvage {entry['strip']} {len(safe) + len(salvage_rejections)}/"
                  f"{len(candidates)}: {len(safe)} accepted", flush=True)

        if len(candidates) == 1:
            salvage_rejections.append({'patch': candidates[0]['patch'], 'reason': initial_failure})
        else:
            midpoint = len(candidates) // 2
            admit(candidates[:midpoint])
            admit(candidates[midpoint:])
        if best is None:
            shutil.copytree(source, destination)
            return {'status': 'fallback_original', 'reason': initial_failure,
                    'salvage_rejections': salvage_rejections}
        temporary, native, validation = best
        write_merged_metadata(source, temporary, native, validation)
        temporary.rename(destination)
        return {'status': 'merged_salvaged', 'initial_failure': initial_failure,
                'salvage_rejections': salvage_rejections, **native, 'validation': validation}


def save_plan(path, plan):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(plan, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument('--patch-root', type=Path, default=DEFAULT_PATCH_ROOT)
    parser.add_argument('--plan-output', type=Path, required=True)
    parser.add_argument('--merge-output', type=Path, help='Write a new complete strip collection here')
    parser.add_argument('--executable', type=Path, default=Path(__file__).resolve().parent.parent /
                        'volume-cartographer/build/bin/anchored_strip_merge')
    parser.add_argument('--strip', help='Process only this strip basename')
    parser.add_argument('--limit-strips', type=int)
    parser.add_argument('--tolerance', type=float, default=2)
    parser.add_argument('--min-overlap-fraction', type=float, default=0.2)
    parser.add_argument('--min-overlap-points', type=int, default=64)
    parser.add_argument('--min-new-points', type=int, default=32)
    parser.add_argument('--query-stride', type=int, default=2)
    parser.add_argument('--merge-workers', type=int, default=1,
                        help='Merge this many independent strips at once after discovery is complete')
    args = parser.parse_args()
    if (not 0 < args.min_overlap_fraction <= 1 or args.tolerance <= 0
            or args.query_stride < 1 or args.merge_workers < 1):
        parser.error('Invalid overlap settings')
    if args.merge_output and not args.executable.is_file():
        parser.error(f'Merger executable does not exist: {args.executable}')
    candidates = patch_catalog(args.patch_root / 'graph_patches_only')
    available = {**patch_catalog(args.patch_root / 'graph_patches_merged'), **candidates}
    manifest = json.loads((args.outputs / 'fiber_merge_all_strips' / 'manifest.json').read_text())
    rows = manifest['strips']
    if args.strip:
        rows = [row for row in rows if row['name'] == args.strip]
        if not rows:
            parser.error(f'Unknown strip: {args.strip}')
    if args.limit_strips is not None:
        rows = rows[:args.limit_strips]
    plan = {'type': 'anchored_strip_extension_plan', 'version': 2,
            'settings': {'tolerance': args.tolerance, 'min_overlap_fraction': args.min_overlap_fraction,
                         'min_overlap_points': args.min_overlap_points,
                         'min_new_points': args.min_new_points, 'query_stride': args.query_stride},
            'strips': []}
    if args.plan_output.exists():
        previous = json.loads(args.plan_output.read_text())
        if (previous.get('type') != plan['type'] or previous.get('version') != plan['version']
                or previous.get('settings') != plan['settings']):
            parser.error('Existing plan uses different overlap settings; use a new plan path')
        plan = previous
    completed = {row['strip']: row for row in plan['strips']}
    pending = [row for row in rows if row['name'] not in completed]
    if args.merge_workers > 1:
        if not args.merge_output or pending:
            parser.error('Parallel merge requires --merge-output and a complete discovery plan')
        args.merge_output.mkdir(parents=True, exist_ok=True)
        work = [completed[row['name']] for row in rows
                if 'merge' not in completed[row['name']]
                or not (args.merge_output / 'strips' / row['name']).exists()]
        for entry in work:
            destination = args.merge_output / 'strips' / entry['strip']
            if destination.exists():
                raise ValueError(f'Output exists without a saved merge report: {destination}')
        with (args.merge_output / '.merge.lock').open('w') as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error('Another merge run is still active for this output directory')
            with ThreadPoolExecutor(max_workers=args.merge_workers) as pool:
                futures = {pool.submit(merge_strip, entry, args.outputs, args.merge_output,
                                       args.executable, args.tolerance): entry for entry in work}
                for number, future in enumerate(as_completed(futures), 1):
                    entry = futures[future]
                    destination = args.merge_output / 'strips' / entry['strip']
                    try:
                        entry['merge'] = future.result()
                    except Exception as error:
                        if not destination.exists():
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            source = args.outputs / 'fiber_merge_all_strips' / 'strips_flattened' / entry['strip']
                            shutil.copytree(source, destination)
                        entry['merge'] = {'status': 'fallback_original', 'reason': str(error)}
                    save_plan(args.plan_output, plan)
                    print(f"[{number}/{len(work)}] {entry['strip']}: "
                          f"{entry['merge']['status']} ({len(entry['candidates'])} candidates)",
                          flush=True)
        return
    if pending:
        index, positions, empty = build_shared_index(available, pending, args.outputs,
                                                     args.tolerance)
        if empty:
            print(f'Skipped {len(empty)} patches without valid quads', flush=True)
    else:
        index, positions = None, None
    candidate_surfaces = ({positions['graph', patch_id]: patch_id for patch_id in candidates
                           if ('graph', patch_id) in positions} if positions else {})
    nearby_cache = {}
    for i, row in enumerate(rows, 1):
        entry = completed.get(row['name'])
        if entry is None:
            entry = scan_strip(row, args.outputs, candidates, index, positions,
                               candidate_surfaces, nearby_cache, args)
            plan['strips'].append(entry)
            completed[row['name']] = entry
            save_plan(args.plan_output, plan)
        destination = args.merge_output / 'strips' / entry['strip'] if args.merge_output else None
        if args.merge_output and ('merge' not in entry or not destination.exists()):
            if destination.exists():
                raise ValueError(f'Output exists without a saved merge report: {destination}')
            try:
                entry['merge'] = merge_strip(entry, args.outputs, args.merge_output,
                                             args.executable, args.tolerance)
            except Exception as error:
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source = args.outputs / 'fiber_merge_all_strips' / 'strips_flattened' / entry['strip']
                    shutil.copytree(source, destination)
                entry['merge'] = {'status': 'fallback_original', 'reason': str(error)}
            save_plan(args.plan_output, plan)
        print(f"[{i}/{len(rows)}] {entry['strip']}: {len(entry['candidates'])} candidates "
              f"from {entry['nearby_candidates']} nearby patches; "
              f"{entry.get('merge', {}).get('status', 'planned')}", flush=True)


if __name__ == '__main__':
    main()
