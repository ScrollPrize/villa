"""CPU-only spatial preview of H/V classification changes in source curves."""
import json
from pathlib import Path

import numpy as np

from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import (
    FiberBlockIndex, fiber_catalog, fiber_volume, read_polylines,
)
from vesuvius.neural_tracing.tracers.fibers.evaluation.monitor.microbatch import clipped_polyline
from .fiber_normal_sign_visualization import obj_fiber


GROUPS = dict(h=lambda old, new: new == 0, v=lambda old, new: new == 1,
              broken=lambda old, new: new == 2,
              h_that_were_v=lambda old, new: (old == 1) & (new == 0),
              v_that_were_h=lambda old, new: (old == 0) & (new == 1))


def random_groups(ids, original, labels, count, seed):
    """Independent uniform samples per group; no length or spatial selection bias."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('sample_per_group must be a positive integer')
    rng = np.random.default_rng(seed)
    return {name: sorted(rng.choice(ids[include(original, labels)],
                 size=min(count, int(np.count_nonzero(include(original, labels)))), replace=False).tolist())
            for name, include in GROUPS.items()}


def candidate_boxes(catalog, ids, original, labels, size):
    """Rank fixed spatial cells by balanced counts of both change directions."""
    changed = (labels != 2) & (original != labels)
    directions = dict(zip(ids[changed], labels[changed]))
    rows = [np.r_[(np.asarray(r['bbox'][:3])+r['bbox'][3:])/2, directions[r['id']]]
            for r in catalog if r['id'] in directions]
    if not rows:
        raise ValueError('No changed fibers')
    rows = np.asarray(rows)
    cells, inverse = np.unique(np.floor(rows[:, :3] / size).astype(np.int64), axis=0, return_inverse=True)
    counts = np.stack([np.bincount(inverse, weights=rows[:, 3] == label, minlength=len(cells))
                       for label in (0, 1)], axis=1)
    order = sorted(range(len(cells)), key=lambda i: (-min(counts[i]), -sum(counts[i]), tuple(cells[i])))
    return [(cells[i] * size, (cells[i] + 1) * size) for i in order]


def crop(index, fibers, ids, original, labels, lower, upper):
    selected = sorted(index.query(np.asarray([(lower+upper)/2]), (upper-lower)/2))
    records = []
    for row, points in read_polylines(fibers, selected):
        index = int(np.searchsorted(ids, row['id']))
        if index == len(ids) or ids[index] != row['id']:
            raise ValueError('Source/solution fiber IDs do not match')
        segments = clipped_polyline(points, lower, upper).reshape(-1, 3, 3)[:, :2]
        segments = segments[np.any(segments[:, 0] != segments[:, 1], axis=1)]
        if len(segments):
            records.append((row['id'], int(original[index]), int(labels[index]), segments))
    return records


def export(root, records, lower, upper, *, selections=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    colors = dict(h=(0.1, 0.4, 1), v=(1, 0.5, 0.1), broken=(0.5, 0.5, 0.5),
                  h_that_were_v=(0.1, 0.9, 0.4), v_that_were_h=(0.9, 0.1, 0.7))
    (root / 'fibers.mtl').write_text(''.join(
        f'newmtl {name}\nKd {r} {g} {b}\n' for name, (r, g, b) in colors.items()))
    result = dict(bounds_xyz=[np.asarray(lower).tolist(), np.asarray(upper).tolist()],
                  coordinates='native L0 XYZ base voxels; no translation or rescaling', groups={})
    for name, include in GROUPS.items():
        first = 1
        selected = []
        selected_ids = None if selections is None else set(selections[name])
        with (root / f'{name}.obj').open('w') as stream:
            stream.write('# Original fiber curves; pairwise line elements, no faces\nmtllib fibers.mtl\n')
            for fid, old, new, segments in records:
                if not include(old, new) or (selected_ids is not None and fid not in selected_ids):
                    continue
                selected.append(fid)
                for part, points in enumerate(segments):
                    stream.write(obj_fiber(f'{fid}_segment_{part}', points, first, new == 2, name))
                    first += 2
        result['groups'][name] = dict(fibers=len(selected), fiber_ids=selected,
                                      vertices=first-1, lines=(first-1)//2)
    result['records'] = [dict(fiber_id=fid, original_label=('H','V')[old],
                              label=('H','V','broken')[new], lines=len(segments))
                         for fid, old, new, segments in records]
    return result


def run(config):
    fibers = Path(config['fibers']).resolve()
    with np.load(config['comparison'], allow_pickle=False) as data:
        ids, original, labels = (data[k] for k in ('ids', 'original_labels', 'labels'))
    if (ids.ndim != 1 or len(ids) == 0 or not np.all(ids[1:] > ids[:-1])
            or original.shape != ids.shape or labels.shape != ids.shape
            or not np.isin(original, (0, 1)).all() or not np.isin(labels, (0, 1, 2)).all()):
        raise ValueError('Invalid comparison arrays')
    catalog = fiber_catalog(fibers, fiber_volume(fibers))
    if not np.array_equal([r['id'] for r in catalog], ids):
        raise ValueError('Source/comparison fiber IDs do not match')
    if 'sample_per_group' in config:
        if 'bounds_xyz' in config:
            raise ValueError('Full-fiber sampling cannot be combined with crop bounds')
        selections = random_groups(ids, original, labels, config['sample_per_group'], config.get('seed', 0))
        selected = sorted({fid for group in selections.values() for fid in group})
        records = []
        lower = np.full(3, np.inf)
        upper = np.full(3, -np.inf)
        for row, points in read_polylines(fibers, selected):
            index = int(np.searchsorted(ids, row['id']))
            if index == len(ids) or ids[index] != row['id']:
                raise ValueError('Source/solution fiber IDs do not match')
            lower = np.minimum(lower, points.min(axis=0))
            upper = np.maximum(upper, points.max(axis=0))
            records.append((row['id'], int(original[index]), int(labels[index]),
                            np.stack([points[:-1], points[1:]], axis=1)))
        result = export(config['output_root'], records, lower, upper, selections=selections)
        result.update(full_fibers=True, sample_per_group=config['sample_per_group'], seed=config.get('seed', 0),
                      sampling='independent uniform random samples per group, without replacement')
        return finish(config, fibers, result)
    if 'bounds_xyz' in config:
        boxes = [np.asarray(config['bounds_xyz'], dtype=float)]
    else:
        size = float(config.get('size_base_voxels', 2048))
        if not np.isfinite(size) or size <= 0:
            raise ValueError('Crop size must be positive')
        boxes = candidate_boxes(catalog, ids, original, labels, size)
    spatial_index = FiberBlockIndex(fibers, ids)
    for lower, upper in boxes[:20]:
        if not np.isfinite([lower, upper]).all() or np.any(upper <= lower):
            raise ValueError('Invalid crop bounds')
        records = crop(spatial_index, fibers, ids, original, labels, lower, upper)
        changes = [sum(old == a and new == b for _, old, new, _ in records)
                   for a, b in ((0, 1), (1, 0))]
        print(json.dumps(dict(bounds_xyz=[lower.tolist(), upper.tolist()],
                              fibers=len(records), changes_hv_vh=changes)), flush=True)
        if 'bounds_xyz' in config or min(changes) >= config.get('minimum_each_change', 3):
            break
    else:
        raise ValueError('No candidate crop has enough changed fibers in both directions')
    result = export(config['output_root'], records, lower, upper)
    return finish(config, fibers, result)


def finish(config, fibers, result):
    result['inputs'] = dict(fibers=str(fibers), comparison=str(Path(config['comparison']).resolve()))
    (Path(config['output_root']) / 'index.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'fiber_ids'}
                      for k, v in result['groups'].items()}), flush=True)
    return result
