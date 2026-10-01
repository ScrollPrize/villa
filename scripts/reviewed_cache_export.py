"""Published-only, portable cache exports for the reviewed-reference comparison."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from reviewed_fiber_evaluation import ROOT, REPO, validate_result
from lasagna.inference_provenance import atomic_write
from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import (
    _canonical_source_label, _export_channel_subset, export_union_lasagna_manifests,
)

CHANNELS = ('fiber_presence.ome.zarr', 'fiber_nx.ome.zarr', 'fiber_ny.ome.zarr')


def published_keys(array):
    """Only numeric z/y/x files, never hidden writer or accumulator directories."""
    result = {}
    with os.scandir(array) as zs:
        for z in zs:
            if not z.name.isdecimal() or not z.is_dir(follow_symlinks=False):
                continue
            with os.scandir(z.path) as ys:
                for y in ys:
                    if not y.name.isdecimal() or not y.is_dir(follow_symlinks=False):
                        continue
                    with os.scandir(y.path) as xs:
                        for x in xs:
                            if x.name.isdecimal() and x.is_file(follow_symlinks=False):
                                result[(int(z.name), int(y.name), int(x.name))] = x.stat().st_size
    return result


def export_source(args):
    root, output = args
    metadata = json.loads((root / 'inference.json').read_text())
    label = _canonical_source_label(metadata['source']['locator'].rstrip('/'))
    manifest = json.loads((root / 'inference.lasagna.json').read_text())
    paths = {v['zarr'] for v in manifest['groups'].values()}
    levels = {Path(p).name for p in paths}
    if len(levels) != 1 or not next(iter(levels)).isdecimal():
        raise ValueError(f'unsupported level layout: {root}')
    level = int(next(iter(levels)))
    if level not in (1, 3):
        raise ValueError(f'unexpected evaluated-model level: {level}')
    if paths != {f'{c}/{level}' for c in CHANNELS}:
        raise ValueError(f'unexpected channel manifest: {root}')
    arrays = [json.loads((root / c / str(level) / '.zarray').read_text()) for c in CHANNELS]
    if any(a.get('dimension_separator', '.') != '/' for a in arrays):
        raise ValueError('export requires slash-separated chunk layout')
    for channel in CHANNELS:
        base = root / channel
        if (base/'.zattrs').exists():
            attrs = json.loads((base/'.zattrs').read_text())
            if any(d['path'] != str(level) for m in attrs.get('multiscales', []) for d in m['datasets']):
                raise ValueError('channel metadata advertises unexported levels')
        if (base/'.zmetadata').exists():
            consolidated = json.loads((base/'.zmetadata').read_text())['metadata']
            if any('/' in k and k.split('/')[0] != str(level) for k in consolidated):
                raise ValueError('consolidated metadata advertises unexported levels')
    with ThreadPoolExecutor(max_workers=3) as pool:
        inventories = list(pool.map(published_keys, [root / c / str(level) for c in CHANNELS]))
    keys = set.intersection(*(set(i) for i in inventories))
    incomplete = len(set.union(*(set(i) for i in inventories)) - keys)
    destination = output / 'vc3d_channels' / label
    # The export is append-free: a second run must use the same frozen inventory.
    inventory_path = output / 'inventories' / f'{label}.json'
    inventory = dict(source_root=str(root), level=level, keys=sorted(keys))
    encoded = json.loads(json.dumps(inventory))
    if inventory_path.exists() and json.loads(inventory_path.read_text()) != encoded:
        raise ValueError(f'cache changed; choose a new export version: {inventory_path}')
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(inventory_path, inventory, durable=False)
    print(f'export {output.name}/{label}: {len(keys)} complete chunks; {incomplete} incomplete excluded', flush=True)
    _export_channel_subset(root, destination, keys, CHANNELS, chunk_mode='symlink', level=level)
    return dict(source_id=label, cache_root=str(destination), subset=True,
                source_root=str(root), level=level, chunk_count=len(keys),
                incomplete_chunks_excluded=incomplete,
                channel_bytes={c: sum(i[k] for k in keys) for c, i in zip(CHANNELS, inventories)})


def run(model=None):
    if model not in (None, 'halfsigma_current_raw_v2'):
        raise ValueError('unsupported cache export model')
    output = ROOT / ('cache_sync_u31200_v1' if model else 'cache_sync_v1')
    rows = []
    for model in ((model,) if model else ('lowres', 'fullres', 'gaussian')):
        validate_result(json.loads((ROOT/model/'full/summary.json').read_text()))
        summary = json.loads((ROOT / model / 'full/inference_summary.json').read_text())
        roots = sorted({Path(p) for p in summary['cache_roots'].values()})
        labels = [_canonical_source_label(json.loads((r/'inference.json').read_text())['source']['locator'].rstrip('/')) for r in roots]
        if len(set(labels)) != len(labels):
            raise ValueError('multiple cache roots for same scroll; resolve before exporting')
        target = output / model
        with ThreadPoolExecutor(max_workers=3) as pool:
            entries = list(pool.map(export_source, [(r, target) for r in roots]))
        atomic_write(target / 'manifest.json', dict(vc3d_channels=entries), durable=False)
        export_union_lasagna_manifests(target)
        for entry in entries:
            row = dict(model=model, **entry)
            row['chunk_bytes'] = sum(row['channel_bytes'].values())
            # Count logical bytes, dereferencing every chunk once, not disk blocks.
            dest = Path(row['cache_root'])
            metadata_bytes = 0
            linked_files = 0
            for parent, dirs, files in os.walk(dest):
                if any(d.startswith('.') for d in dirs):
                    raise ValueError(f'unexpected hidden directory in export: {parent}')
                for name in files:
                    p = Path(parent) / name
                    if p.is_symlink():
                        if not p.is_file():
                            raise ValueError(f'broken chunk link: {p}')
                        linked_files += 1
                    else:
                        metadata_bytes += p.stat().st_size
            if linked_files != row['chunk_count'] * 3:
                raise ValueError('export link count mismatch')
            row['transfer_bytes'] = row['chunk_bytes'] + metadata_bytes
            rows.append(row)
            print(f"verified {model}/{row['source_id']}: {row['transfer_bytes']/2**30:.3f} GiB", flush=True)
    atomic_write(output/'export_report.json', dict(
        scope='All currently published chunks, including earlier reused coverage; not only new evaluation chunks.',
        evaluation_root=str(ROOT), rows=rows,
        transfer_bytes=sum(r['transfer_bytes'] for r in rows)), durable=False)
    print(f'COMPLETE {output}', flush=True)


if __name__ == '__main__':
    control = json.loads((REPO/'scripts/training_failure_mining_control.json').read_text())
    run(control.get('cache_export_model'))
