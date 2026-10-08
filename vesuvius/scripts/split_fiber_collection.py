"""Reproducible whole-fiber train/test splits with per-source length quotas."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from vc3d_fiber_format import parse_vc3d_fiber_format

SOURCES = (
    'PHerc0139', 'PHerc0332', 'PHerc1229', 'PHerc1451',
    'PHercParis4_eval_fibers', 'PHercParis4_fibers',
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fiber_length(points) -> float:
    length = math.fsum(math.dist(a, b) for a, b in zip(points, points[1:]))
    if not math.isfinite(length) or length <= 0:
        raise ValueError('Fiber must have positive finite polyline length')
    return length


def assign_source(records, *, seed: int, source: str, fraction: float):
    if not 0 < fraction < 1:
        raise ValueError('test fraction must lie strictly between zero and one')
    if len(records) < 2:
        raise ValueError(f'{source}: need at least two fibers')
    ordered = sorted(records, key=lambda r: (
        digest(json.dumps([seed, source, r['path']], separators=(',', ':')).encode()),
        r['path'],
    ))
    total = math.fsum(r['length_base_voxels'] for r in ordered)
    target = fraction * total
    chosen = []
    for rank, record in enumerate(ordered):
        record['random_rank'] = rank
        if math.fsum(chosen) < target:
            record['split'] = 'test'
            chosen.append(record['length_base_voxels'])
        else:
            record['split'] = 'train'
    if len(chosen) == len(ordered):
        raise ValueError(f'{source}: random prefix leaves no training fibers')
    test_length = math.fsum(chosen)
    assert test_length >= target and math.fsum(chosen[:-1]) < target
    return {
        'source': source, 'files': len(ordered),
        'train_files': len(ordered) - len(chosen), 'test_files': len(chosen),
        'total_length': total, 'test_target_length': target,
        'test_length': test_length, 'train_length': total - test_length,
        'test_length_percent': 100 * test_length / total,
        'test_prefix_before_last_length': math.fsum(chosen[:-1]),
    }


def plan_split(root: Path, *, seed=20260910, fraction=0.05):
    root = root.resolve()
    expected = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        sha, name = line.split('  ', 1)
        expected[name] = sha
    records, summaries = [], []
    for source in SOURCES:
        local = []
        for path in sorted((root / source).rglob('*.json')):
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            sha = digest(data)
            if expected.get(relative) != sha:
                raise ValueError(f'Snapshot checksum mismatch: {relative}')
            obj = json.loads(data)
            fiber = parse_vc3d_fiber_format(obj, path=path)
            points = fiber.line_points_xyz
            local.append({
                'path': relative, 'source': source, 'sha256': sha,
                'bytes': len(data), 'line_points': len(points),
                'length_base_voxels': fiber_length(points),
                'geometry_sha256': digest(json.dumps(min(points, points[::-1]), separators=(',', ':')).encode()),
                'branch_files': [b['branch_file'] for b in obj.get('branches', []) if b.get('branch_file')],
            })
        summaries.append(assign_source(local, seed=seed, source=source, fraction=fraction))
        records.extend(local)
        print(f'Planned {source}: {len(local)} fibers', flush=True)
    if {r['path'] for r in records} != set(expected):
        raise ValueError('Source file inventory does not match snapshot SHA256SUMS')
    groups = {}
    by_path = {r['path']: r for r in records}
    cross_branches = []
    for r in records:
        scroll = 'PHercParis4' if r['source'].startswith('PHercParis4') else r['source']
        groups.setdefault((scroll, r['geometry_sha256']), []).append(r)
        for branch in r['branch_files']:
            target = os.path.normpath(Path(r['path']).parent / branch)
            other = by_path.get(target)
            if other and other['split'] != r['split']:
                cross_branches.append([r['path'], target])
    return {
        'schema': 1, 'source_root': str(root), 'seed': seed,
        'test_fraction': fraction,
        'algorithm': 'sha256(seed, source, relative_path) ascending; minimal whole-file prefix reaching test_fraction * summed polyline length',
        'length_units': 'source base-voxel coordinates; sums are only compared within each source',
        'snapshot_inventory_sha256': digest((root / 'SHA256SUMS').read_bytes()),
        'source_documentation_sha256': digest((root / 'SOURCES.md').read_bytes()),
        'sources': summaries, 'files': sorted(records, key=lambda r: r['path']),
        'duplicate_geometry_groups': [[r['path'] for r in rs] for rs in groups.values() if len(rs) > 1],
        'cross_split_duplicate_geometry_groups': [[r['path'] for r in rs] for rs in groups.values() if len({r['split'] for r in rs}) > 1],
        'cross_split_branch_references': cross_branches,
    }


def summary_markdown(plan):
    percent = 100 * plan['test_fraction']
    lines = [
        f'# Per-source {100 - percent:g}/{percent:g} fiber-length split', '',
        f"Seed: `{plan['seed']}`. Whole fibers in deterministic pseudorandom order; test gets the shortest prefix reaching at least {percent:g}% of each source's polyline length. No fiber is cut.", '',
        'Lengths sum Euclidean distances between consecutive line_points in base voxel coordinates. Each Paris4 collection is split independently. Symlinks retain original bytes; source data must remain available.', '',
        '| Source | Train files | Test files | Train length | Test length | Test length % |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in plan['sources']:
        lines.append(f"| {row['source']} | {row['train_files']} | {row['test_files']} | {row['train_length']:.3f} | {row['test_length']:.3f} | {row['test_length_percent']:.4f}% |")
    lines.extend(['', 'This is a fiber-level split, not a spatially disjoint or branch-connected-component split. Nearby/intersecting fibers can occur in both sets; branch references remain untouched and must not be followed to expand the selected training files.', '',
                  f"Cross-split branch references: {len(plan['cross_split_branch_references'])}. Cross-split identical geometry groups: {len(plan['cross_split_duplicate_geometry_groups'])}.", '',
                  'manifest.json records every file, original checksum, length, random rank and membership. SOURCES.md copies download provenance. Rerunning the same source snapshot and seed yields the same membership.', ''])
    return '\n'.join(lines)


def publish(plan, output: Path):
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite {output}')
    root = Path(plan['source_root'])
    metadata = {}
    for name, key in [('SHA256SUMS', 'snapshot_inventory_sha256'), ('SOURCES.md', 'source_documentation_sha256')]:
        data = (root / name).read_bytes()
        if digest(data) != plan[key]:
            raise ValueError(f'Source provenance changed after planning: {name}')
        metadata[name] = data
    # Staging lives beside output so relative links survive atomic directory rename.
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}.', dir=output.parent) as temp:
        staging = Path(temp)
        for record in plan['files']:
            source = root / record['path']
            if digest(source.read_bytes()) != record['sha256']:
                raise ValueError(f'Source changed after planning: {source}')
            target = staging / record['split'] / record['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.path.relpath(source, target.parent))
        (staging / 'manifest.json').write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
        (staging / 'README.md').write_text(summary_markdown(plan))
        (staging / 'SOURCES.md').write_bytes(metadata['SOURCES.md'])
        (staging / 'SOURCE_SHA256SUMS').write_bytes(metadata['SHA256SUMS'])
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--test-fraction', type=float, default=0.05)
    parser.add_argument('--apply', action='store_true', help='Publish split; default only prints plan')
    args = parser.parse_args()
    plan = plan_split(args.source, seed=args.seed, fraction=args.test_fraction)
    print(summary_markdown(plan), flush=True)
    if args.apply:
        publish(plan, args.output)
        print(f'Published {args.output}', flush=True)


if __name__ == '__main__':
    main()
