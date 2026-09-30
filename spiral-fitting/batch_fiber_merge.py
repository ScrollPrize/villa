"""Export one final continuous TIFXYZ per assigned fiber, with resumable logs.

Uses prototype_fiber_merge --continuous-only and the shared preview validator.
No fitting logic lives here. See prototype_fiber_merge.md for build/run steps.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

import numpy as np
from preview_fiber_merge import load_surface, surface_metrics


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def fingerprint(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _is_canonical_uuid(value):
    try:
        return isinstance(value, str) and str(uuid.UUID(value)) == value
    except ValueError:
        return False


def finalize_segment_ids_locked(root):
    """Caller holds .batch.lock; repair IDs only after all export attempts finish."""
    status = json.loads((root / 'status.json').read_text())
    if not status.get('complete') or any(row['status'] in ('pending', 'running') for row in status['fibers']):
        raise ValueError('Exports are not finished; metadata was not changed')
    expected = {root / row['surface'] / 'meta.json' for row in status['fibers'] if row['status'] == 'exported'}
    actual = set((root / 'strips').glob('*/meta.json'))
    if actual != expected:
        raise ValueError('Published metadata files do not match the completed export manifest')
    # Each exported strip is a new object and needs its own random uuid. Keep
    # one the fitter already minted (so the pass is idempotent); replace a
    # missing, non-canonical or batch-duplicated one (e.g. every strip sharing
    # the fitter's intermediate "refined" name) with a fresh uuid4.
    plan = []
    seen = set()
    for path in sorted(actual):
        metadata = json.loads(path.read_text())
        if metadata.get('type') != 'seg' or metadata.get('format') != 'tifxyz':
            raise ValueError(f'Unexpected segment metadata: {path}')
        identifier = metadata.get('uuid')
        if not _is_canonical_uuid(identifier) or identifier in seen:
            identifier = str(uuid.uuid4())
        seen.add(identifier)
        plan.append((path, identifier, metadata))
    backup_root = root / 'metadata_before_unique_ids'
    changes = []
    for path, identifier, metadata in plan:
        backup = backup_root / path.parent.name / 'meta.json'
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        original = json.loads(backup.read_text())
        if metadata.get('uuid') != identifier:
            metadata['uuid'] = identifier
            write_json(path, metadata)
        changes.append({'path': str(path.relative_to(root)), 'previous_uuid': original.get('uuid'), 'uuid': identifier})
    verified = [json.loads(path.read_text())['uuid'] for path, _, _ in plan]
    if verified != [identifier for _, identifier, _ in plan] or len(set(verified)) != len(plan):
        raise ValueError('Segment ID verification failed')
    result = {'complete': True, 'segment_count': len(plan), 'unique_ids': len(set(verified)),
              'changes': changes, 'backup_directory': str(backup_root.relative_to(root))}
    write_json(root / 'metadata_ids_status.json', result)
    return result


def check_surface(root, info):
    if not info.get('valid') or info.get('component_count') != 1:
        raise ValueError(info.get('message', 'Expected one valid continuous component'))
    xyz, _, _ = load_surface(root, info)
    if not np.isfinite(xyz[xyz[..., 0] != -1]).all():
        raise ValueError('Nonfinite XYZ in exported surface')
    metrics = surface_metrics(xyz)
    if metrics['quad_connected_regions'] != 1 or metrics['valid_quads'] < 2 or metrics['opposing_triangle_quads']:
        raise ValueError(f'Continuous surface failed validation: {metrics}')
    metadata = json.loads((root / info['path'] / 'meta.json').read_text())
    if not metadata['constraint_graph']['includes_fiber_ribbons']:
        raise ValueError('Export did not include the fiber strip')
    return metrics


def run_one(name, patch_count, config):
    started = time.monotonic()
    root = Path(config['output'])
    job = root / 'jobs' / Path(name).stem
    job.mkdir(parents=True, exist_ok=True)
    number = 1
    while (job / f'attempt_{number:03d}').exists() or (job / f'attempt_{number:03d}.log').exists():
        number += 1
    attempt = job / f'attempt_{number:03d}'
    log = attempt.with_suffix('.log')
    row = {'fiber': name, 'patch_count': patch_count, 'status': 'running',
           'log': str(log.relative_to(root)), 'report': str((attempt / 'report.json').relative_to(root))}
    write_json(job / 'status.json', row)
    try:
        fiber_only = config.get('fiber_only', False)
        command = [config['executable'], str(root / 'links.json'), name, str(attempt),
                   str(config['spacing']), str(0 if fiber_only else config['sweeps']),
                   '--fiber-only' if fiber_only else '--continuous-only']
        if not fiber_only and config.get('shape_weight', 1) != 1:
            command.extend(['--shape-weight', str(config['shape_weight'])])
        environment = dict(os.environ, AGENTS_AGENT_MODE='1', OMP_NUM_THREADS='1',
                           VC3D_CONSTRAINT_FIT_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with log.open('w') as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        row['returncode'] = process.returncode
        report = json.loads((attempt / 'report.json').read_text()) if (attempt / 'report.json').exists() else {}
        info = report.get('continuous', {}).get('refined', {})
        if process.returncode or not report.get('complete') or not info.get('valid'):
            tail = '\n'.join(log.read_text(errors='replace').splitlines()[-5:])
            raise ValueError(info.get('message') or f'Fitter exit {process.returncode}: {tail}')
        row['metrics'] = check_surface(attempt, info)
        if fiber_only:
            metadata = json.loads((attempt / info['path'] / 'meta.json').read_text())
            if not metadata['constraint_graph'].get('fiber_only'):
                raise ValueError('Expected a fiber-only strip')
        destination = root / 'strips' / (Path(name).stem + '.tifxyz')
        if destination.exists():
            raise ValueError(f'Refusing to overwrite existing surface: {destination}')
        shutil.move(attempt / info['path'], destination)
        relative = os.path.relpath(destination, attempt)
        info['path'] = relative
        for component in info['components']:
            component['path'] = relative
        write_json(attempt / 'report.json', report)
        row.update(status='exported', surface=str(destination.relative_to(root)),
                   export_kind='fiber_only' if fiber_only else 'merged',
                   dropped_single_quad_components=info['dropped_single_quad_components'],
                   dropped_other_components=info['dropped_other_components'],
                   failed_fit_attempts=sum(not fit['valid'] for fit in report.get('fits', [])))
    except Exception as error:
        row.update(status='failed', message=str(error))
    row['seconds'] = round(time.monotonic() - started, 3)
    write_json(job / 'status.json', row)
    return row


def run_batch(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Linux/macOS advisory lock is automatically released if the runner exits.
    with (root / '.batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(args, root)


def run_locked(args, root):
    executable = args.executable.resolve()
    document = json.loads(args.links.read_text())
    if document.get('type') != 'vc_fiber_patch_links' or document.get('version') != 1:
        raise ValueError('Expected vc_fiber_patch_links v1')
    config = {'executable': str(executable), 'executable_sha256': fingerprint(executable),
              'input_sha256': fingerprint(args.links), 'spacing': args.spacing,
              'sweeps': args.sweeps, 'output': str(root)}
    if getattr(args, 'shape_weight', 1) != 1:
        config['shape_weight'] = args.shape_weight
    if getattr(args, 'fiber_only', False):
        config['fiber_only'] = True
    for library in ('libvc_constraints.so', 'libvc_constraints.dylib'):
        path = executable.parent.parent / 'core' / library
        if path.exists():
            config['constraints_library_sha256'] = fingerprint(path)
    manifest = root / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != config:
            raise ValueError('Input, executable, library, or fitting settings differ from this batch; use a new output directory')
    else:
        if any(root.iterdir()) and set(p.name for p in root.iterdir()) != {'.batch.lock'}:
            raise ValueError('Output directory is not an existing batch or empty directory')
        shutil.copyfile(args.links, root / 'links.json')
        write_json(manifest, config)
    if fingerprint(root / 'links.json') != config['input_sha256']:
        raise ValueError('Batch input snapshot changed')
    (root / 'strips').mkdir(exist_ok=True)
    rows = {}
    pending = []
    for name, fiber in sorted(document['fibers'].items()):
        if Path(name).name != name or not name.endswith('.json'):
            raise ValueError(f'Invalid fiber basename: {name}')
        count = len({entry['patch'] for entry in fiber['linked']})
        if config.get('fiber_only') and fiber['direction'] != 'H':
            raise ValueError(f'Fiber-only batch contains a nonhorizontal fiber: {name}')
        status_path = root / 'jobs' / Path(name).stem / 'status.json'
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        if not count and not config.get('fiber_only'):
            rows[name] = {'fiber': name, 'patch_count': 0, 'status': 'no_assignments'}
        elif previous.get('status') == 'exported':
            if not all((root / previous['surface'] / filename).exists() for filename in ('x.tif', 'y.tif', 'z.tif', 'meta.json')):
                raise ValueError(f'Previously exported surface is missing: {name}')
            rows[name] = previous
        elif previous.get('status') == 'failed' and not args.retry_failed:
            rows[name] = previous
        else:
            rows[name] = {'fiber': name, 'patch_count': count, 'status': 'pending'}
            pending.append((name, count))

    def status(complete=False):
        # Per-job files also expose running jobs during long fits.
        counts = dict(Counter(row['status'] for row in rows.values()))
        value = {'complete': complete, 'pid': os.getpid(), 'workers': args.workers,
                 'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'counts': counts, 'fibers': [rows[name] for name in sorted(rows)]}
        write_json(root / 'status.json', value)
        print(json.dumps({'complete': complete, 'counts': counts}), flush=True)

    status()
    # Queue smaller cases first to expose systematic problems early. Scheduling
    # does not change each fiber's deterministic fitting order or thread count.
    pending.sort(key=lambda item: (item[1], item[0]))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, name, count, config): name for name, count in pending}
        for future in as_completed(futures):
            name = futures[future]
            rows[name] = future.result()
            print(f"{rows[name]['status']}: {name} ({rows[name]['seconds']:.1f}s)", flush=True)
            status()
    status(complete=True)
    finalize_segment_ids_locked(root)
    return int(any(row['status'] == 'failed' for row in rows.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('links', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--executable', type=Path, default=Path(__file__).resolve().parent.parent / 'volume-cartographer/build/bin/prototype_fiber_merge')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--spacing', type=float, default=10)
    parser.add_argument('--sweeps', type=int, default=1)
    parser.add_argument('--shape-weight', type=float, default=1,
                        help='Multiplier for per-triangle review shape energy (default: 1)')
    parser.add_argument('--fiber-only', action='store_true',
                        help='Export the generated strip for every horizontal input fiber, without patches')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if (args.workers < 1 or not np.isfinite(args.spacing) or args.spacing <= 0 or args.sweeps < 0 or
        not np.isfinite(args.shape_weight) or args.shape_weight < 1):
        parser.error('workers/spacing must be positive, sweeps nonnegative, and shape weight >= 1')
    if args.fiber_only and args.shape_weight != 1:
        parser.error('--shape-weight does not apply to --fiber-only')
    raise SystemExit(run_batch(args))


if __name__ == '__main__':
    main()
