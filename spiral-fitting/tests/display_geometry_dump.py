#!/usr/bin/env python3
"""Viewer-frame coordinates of Spiral workspace inputs, for regression checks.

The Spiral workspace overlays a fit's patches, point collections and fibers
in the viewer frame: reference-volume voxels, XYZ, at full resolution. This
dump records those coordinates for a fixed set of inputs so the file-backed
workspace (before the object-store port) and the store-backed one can be
compared on the same data:

    # file inputs, as the pre-port workspace read them
    python display_geometry_dump.py files --patch DIR --pcl DOC.json#ID \\
        --fiber FIBER.json [--fiber-scale 4] --out dump.json
    # the same inputs read from an object store (optionally pinned)
    python display_geometry_dump.py store STORE.vcobj --object UUID[@GEOMETRY] \\
        --out dump.json
    python display_geometry_dump.py compare A.json B.json

File mode needs only numpy and tifffile, so it runs unchanged at any
checkout. Patches are their TIFXYZ grids as the workspace displays them: a
vertex is invalid (-1) when stored so, or when its ``mask.tif`` block holds a
sample below 255, as VC3D's tifxyz load applies the mask; the store keeps that
mask as its ``valid`` channel. PCL points are their documents' ``p``;
version-3 fiber JSONs carry no
frame, so ``--fiber-scale`` gives their stored-per-reference scale (the
acceptance dataset's are at 4). Store mode divides stored XYZ by each
object's declared ``scale_factor``, as the store's geometryScaleToVolume
does. Objects are keyed by the store's derivation names (``patch:DIR``,
``point_collection:DOC:ID``, ``fiber:STEM``), so both modes key one input
alike.

Each entry records counts, invalid and masked vertices, bounds, a SHA-256
of the float32 coordinates and, for point collections, every point by ID.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np


def _digest(xyz):
    return hashlib.sha256(np.ascontiguousarray(xyz, dtype='<f4').tobytes()).hexdigest()


def _summary(xyz, invalid=None):
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    invalid = np.zeros(len(xyz), bool) if invalid is None else np.asarray(invalid).reshape(-1)
    xyz = xyz.copy()
    xyz[invalid] = -1.0
    valid = xyz[~invalid]
    return {
        'count': int(len(xyz)), 'invalid': int(invalid.sum()),
        'min': valid.min(axis=0).tolist() if len(valid) else None,
        'max': valid.max(axis=0).tolist() if len(valid) else None,
        'first': valid[:2].tolist(), 'sha256': _digest(xyz),
    }


def _surface(xyz, valid=None):
    """A (rows, cols, 3) XYZ grid as displayed: a vertex is invalid when all
    of XYZ are -1 or ``valid`` (per vertex) is false."""
    xyz = np.asarray(xyz, dtype=np.float32)
    invalid = np.all(xyz == -1, axis=-1)
    if valid is not None:
        invalid |= ~np.asarray(valid, dtype=bool)
    entry = {'kind': 'surface', 'shape': list(xyz.shape[:2]), **_summary(xyz, invalid)}
    entry['masked'] = None if valid is None else int((~np.asarray(valid, dtype=bool)).sum())
    return entry


def _points(ids, xyz):
    order = np.argsort(np.asarray(ids, dtype=np.int64), kind='stable')
    ids = [int(ids[i]) for i in order]
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)[order]
    return {'kind': 'point_collection', **_summary(xyz),
            'points': {str(i): p.tolist() for i, p in zip(ids, xyz)}}


def _fiber(controls, line):
    return {'kind': 'fiber', 'controls': _summary(controls), 'line': _summary(line)}


# ---------------------------------------------------------------------------
# File inputs
# ---------------------------------------------------------------------------

def _read_tifxyz(directory):
    import tifffile
    xyz = np.stack([tifffile.imread(os.path.join(directory, f'{axis}.tif'))
                    for axis in 'xyz'], axis=-1)
    mask_path = os.path.join(directory, 'mask.tif')
    valid = None
    if os.path.isfile(mask_path):
        mask = tifffile.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        # A mask may be a whole multiple of the grid; a vertex stays valid
        # only when every sample of its block is at least 255.
        rows, cols = xyz.shape[:2]
        block = mask.reshape(rows, mask.shape[0] // rows, cols, mask.shape[1] // cols)
        valid = (block >= 255).all(axis=(1, 3))
    return xyz, valid


def _patch_key(directory_name):
    stem = directory_name[:-len('.tifxyz')] if directory_name.endswith('.tifxyz') else directory_name
    return f'patch:{stem}'


def dump_files(patches, pcls, fibers, fiber_scale):
    entries = {}
    for directory in patches:
        xyz, valid = _read_tifxyz(directory)
        name = os.path.basename(os.path.normpath(directory))
        entries[_patch_key(name)] = _surface(xyz, valid)
    documents = {}
    for spec in pcls:
        path, _, collection = spec.rpartition('#')
        if path not in documents:
            with open(path) as stream:
                documents[path] = json.load(stream)
        points = documents[path]['collections'][collection]['points']
        ids = [int(i) for i in points]
        xyz = [points[i]['p'] for i in points]
        stem = os.path.splitext(os.path.basename(path))[0]
        entries[f'point_collection:{stem}:{collection}'] = _points(ids, xyz)
    for path in fibers:
        with open(path) as stream:
            document = json.load(stream)
        controls = np.array([c['position'] for c in document['control_points']], np.float64)
        line = np.array(document['line_points'], np.float64).reshape(-1, 3)
        stem = os.path.splitext(os.path.basename(path))[0]
        entries[f'fiber:{stem}'] = _fiber(controls / fiber_scale, line / fiber_scale)
    return entries


# ---------------------------------------------------------------------------
# Object-store inputs
# ---------------------------------------------------------------------------

def _array(db, geometry_revision, name):
    descriptor = db.geometry_revision(geometry_revision)
    array = next((a for a in descriptor['arrays'] if a['name'] == name), None)
    if array is None:
        return None
    data = db.read_geometry_array(geometry_revision, name)
    return np.frombuffer(data, dtype=np.dtype(array['dtype'])).reshape(array['shape'])


def _derived_name(record):
    derived = ((record.get('meta') or {}).get('source') or {}).get('uuid_derivation') or {}
    name = derived.get('name') or f'{record["kind"]}:{record["name"]}'
    return _patch_key(name[len('patch:'):]) if name.startswith('patch:') else name


def dump_store(path, objects):
    from vc_object_store import ObjectStore
    entries = {}
    with ObjectStore.open(path, read_only=True) as db:
        for spec in objects:
            uuid, _, pinned = spec.partition('@')
            record = db.object(uuid)
            if record is None:
                raise SystemExit(f'object {uuid} is not in {path}')
            geometry = pinned or db.current_geometry(uuid)['uuid']
            scale = float((record['meta'].get('coordinates') or {})['scale_factor'])
            kind = record['kind']
            if kind == 'surface':
                points = _array(db, geometry, 'points').astype(np.float64)
                invalid = np.all(points == -1, axis=-1)
                xyz = points / scale
                xyz[invalid] = -1
                valid = _array(db, geometry, 'valid')
                entry = _surface(xyz, None if valid is None else valid != 0)
            elif kind == 'point_collection':
                entry = _points(_array(db, geometry, 'point_ids'),
                                _array(db, geometry, 'points') / scale)
            else:
                entry = _fiber(_array(db, geometry, 'control_points') / scale,
                               _array(db, geometry, 'line_points') / scale)
            entry['geometry_revision'] = geometry
            entries[_derived_name(record)] = entry
    return entries


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

_COMPARED = ('kind', 'shape', 'count', 'invalid', 'masked', 'sha256', 'points',
             'controls', 'line')


def compare(a, b):
    """Differences between two dumps, one line each; [] when they agree."""
    problems = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            problems.append(f'{key}: only in {"first" if key in a else "second"}')
            continue
        for field in _COMPARED:
            if a[key].get(field) != b[key].get(field):
                problems.append(f'{key}: {field} differs')
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    modes = parser.add_subparsers(dest='mode', required=True)
    files = modes.add_parser('files')
    files.add_argument('--patch', action='append', default=[])
    files.add_argument('--pcl', action='append', default=[], metavar='DOC.json#ID')
    files.add_argument('--fiber', action='append', default=[])
    files.add_argument('--fiber-scale', type=float, default=1.0)
    files.add_argument('--out', required=True)
    store = modes.add_parser('store')
    store.add_argument('store')
    store.add_argument('--object', action='append', default=[], metavar='UUID[@GEOMETRY]')
    store.add_argument('--out', required=True)
    both = modes.add_parser('compare')
    both.add_argument('first')
    both.add_argument('second')
    args = parser.parse_args(argv)
    if args.mode == 'compare':
        with open(args.first) as a, open(args.second) as b:
            problems = compare(json.load(a)['objects'], json.load(b)['objects'])
        print('\n'.join(problems) or 'identical')
        return 1 if problems else 0
    if args.mode == 'files':
        entries = dump_files(args.patch, args.pcl, args.fiber, args.fiber_scale)
    else:
        entries = dump_store(args.store, args.object)
    with open(args.out, 'w') as stream:
        json.dump({'frame': 'reference-volume voxels, XYZ', 'objects': entries},
                  stream, indent=1, sort_keys=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
