#!/usr/bin/env python3

import argparse
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse.csgraph import minimum_spanning_tree


def load_umbilicus_yx(path):
    with open(path, 'r') as f:
        data = json.load(f)
    pts = sorted(data['control_points'], key=lambda p: p['z'])
    zs = np.array([p['z'] for p in pts], dtype=np.float64)
    yxs = np.array([[p['y'], p['x']] for p in pts], dtype=np.float64)
    return interp1d(zs, yxs, axis=0, fill_value='extrapolate')


def point_xyzs(points):
    # VC point collections store p as [x, y, z].
    return np.array([p['p'] for p in points], dtype=np.float64)


def polar_features(xyz, umbilicus_yx):
    yx = xyz[:, [1, 0]]
    rel_yx = yx - umbilicus_yx(xyz[:, 2])
    theta = np.mod(np.arctan2(rel_yx[:, 0], rel_yx[:, 1]), 2 * np.pi)
    radius = np.linalg.norm(rel_yx, axis=1)
    median_radius = max(float(np.median(radius)), 1.0)

    if len(theta) > 1:
        theta_sorted = np.sort(theta)
        gaps = np.diff(np.r_[theta_sorted, theta_sorted[0] + 2 * np.pi])
        gap_idx = int(np.argmax(gaps))
        seam = (theta_sorted[gap_idx] + float(gaps[gap_idx]) / 2) % (2 * np.pi)
        theta_unwrapped = (theta - seam) % (2 * np.pi)
    else:
        theta_unwrapped = theta.copy()

    # Both coordinates are in voxel-like units: angular displacement becomes local arc length.
    return np.column_stack([radius, theta_unwrapped * median_radius])


def principal_axis(xyz):
    if len(xyz) < 2:
        return np.array([0.0, 0.0, 1.0]), 1.0
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    _, svals, vt = np.linalg.svd(centered, full_matrices=False)
    explained = float(svals[0] ** 2 / np.sum(svals ** 2)) if np.sum(svals ** 2) > 0 else 1.0
    return vt[0], explained


def is_z_oriented(xyz, z_axis_min):
    axis, explained = principal_axis(xyz)
    return abs(float(axis[2])) >= z_axis_min, axis, explained


def mst_diameter_order(features):
    n = len(features)
    if n <= 2:
        return list(range(n))

    diffs = features[:, None, :] - features[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    mst = minimum_spanning_tree(dist).toarray()

    adjacency = [[] for _ in range(n)]
    rows, cols = np.nonzero(mst)
    for i, j in zip(rows.tolist(), cols.tolist()):
        w = float(mst[i, j])
        adjacency[i].append((j, w))
        adjacency[j].append((i, w))

    def farthest_from(start):
        parent = [-1] * n
        acc = [-1.0] * n
        acc[start] = 0.0
        stack = [start]
        while stack:
            u = stack.pop()
            for v, w in adjacency[u]:
                if v == parent[u]:
                    continue
                parent[v] = u
                acc[v] = acc[u] + w
                stack.append(v)
        far = max(range(n), key=lambda i: acc[i])
        return far, parent

    a, _ = farthest_from(0)
    b, parent = farthest_from(a)

    diameter = []
    cur = b
    while cur != -1:
        diameter.append(cur)
        if cur == a:
            break
        cur = parent[cur]
    diameter = diameter[::-1]

    on_diameter = set(diameter)
    order = []
    seen = set()

    def add_branch(u, blocked):
        seen.add(u)
        order.append(u)
        children = [
            (v, w) for v, w in adjacency[u]
            if v != blocked and v not in on_diameter and v not in seen
        ]
        children.sort(key=lambda vw: vw[1])
        for v, _ in children:
            add_branch(v, u)

    prev = -1
    for u in diameter:
        if u not in seen:
            seen.add(u)
            order.append(u)
        branches = [
            (v, w) for v, w in adjacency[u]
            if v != prev and v not in on_diameter and v not in seen
        ]
        branches.sort(key=lambda vw: vw[1])
        for v, _ in branches:
            add_branch(v, u)
        prev = u

    if len(order) != n:
        remaining = [i for i in range(n) if i not in seen]
        remaining.sort(key=lambda i: min(dist[i, j] for j in seen) if seen else 0.0)
        order.extend(remaining)

    ordered_features = features[order]
    # Prefer increasing angular/arc coordinate for non-z-oriented same-wrap strips.
    if ordered_features[-1, 1] < ordered_features[0, 1]:
        order = order[::-1]
    return order


def z_order(xyz, direction):
    order = np.argsort(xyz[:, 2], kind='stable').tolist()
    if direction == 'decreasing':
        order.reverse()
    return order


def default_output_path(input_path):
    path = Path(input_path)
    return str(path.with_name(f'{path.stem}_fixed{path.suffix}'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Reorder each VC point collection independently as a line strip.'
    )
    parser.add_argument('input_json', help='Input VC point-collection JSON.')
    parser.add_argument(
        '-o', '--output',
        help='Output JSON path. Defaults to <input>_fixed.json next to the input.',
    )
    parser.add_argument(
        '--umbilicus',
        default='/home/sean/Documents/volpkgs/s1_ds2.volpkg/umbilicus.json',
        help='Umbilicus JSON used to compute radius/theta for non-z-oriented collections.',
    )
    parser.add_argument(
        '--z-axis-min',
        type=float,
        default=0.75,
        help='Classify a collection as z-oriented when abs(PCA axis z component) is at least this value.',
    )
    parser.add_argument(
        '--z-direction',
        choices=['increasing', 'decreasing'],
        default='increasing',
        help='Direction to use for z-oriented collections.',
    )
    parser.add_argument(
        '--id-start',
        type=int,
        default=1,
        help='First point id to assign in the fixed JSON.',
    )
    parser.add_argument(
        '--preserve-creation-time',
        action='store_true',
        help='Keep original creation_time values. By default creation_time is rewritten to match fixed order.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output or default_output_path(args.input_json)

    with open(args.input_json, 'r') as f:
        data = json.load(f)

    if data.get('vc_pointcollections_json_version') != '1':
        raise ValueError('Input does not look like a VC point-collection JSON version 1 file')

    umbilicus_yx = load_umbilicus_yx(args.umbilicus)
    fixed = deepcopy(data)
    fixed_collections = {}
    next_point_id = int(args.id_start)
    stats = {
        'collections': 0,
        'points': 0,
        'z_oriented_collections': 0,
        'mst_collections': 0,
    }

    for cid, collection in sorted(data.get('collections', {}).items(), key=lambda kv: int(kv[0])):
        old_items = sorted(collection.get('points', {}).items(), key=lambda kv: int(kv[0]))
        old_points = [p for _, p in old_items]
        xyz = point_xyzs(old_points) if old_points else np.zeros((0, 3), dtype=np.float64)

        if len(old_points) <= 2:
            order = list(range(len(old_points)))
            method = 'trivial'
            axis = [0.0, 0.0, 1.0]
            explained = 1.0
        else:
            z_oriented, axis_arr, explained = is_z_oriented(xyz, args.z_axis_min)
            axis = axis_arr.tolist()
            if z_oriented:
                order = z_order(xyz, args.z_direction)
                method = 'z'
                stats['z_oriented_collections'] += 1
            else:
                order = mst_diameter_order(polar_features(xyz, umbilicus_yx))
                method = 'mst'
                stats['mst_collections'] += 1

        new_collection = deepcopy(collection)
        new_points = {}
        for local_order, old_idx in enumerate(order):
            point = deepcopy(old_points[old_idx])
            if not args.preserve_creation_time:
                point['creation_time'] = next_point_id
            new_points[str(next_point_id)] = point
            next_point_id += 1
        new_collection['points'] = new_points
        new_collection.setdefault('metadata', {})
        new_collection['metadata']['fixed_order_method'] = method
        new_collection['metadata']['fixed_order_pca_axis'] = axis
        new_collection['metadata']['fixed_order_pca_explained'] = explained
        fixed_collections[cid] = new_collection

        stats['collections'] += 1
        stats['points'] += len(old_points)

    fixed['collections'] = fixed_collections
    fixed.setdefault('metadata', {})
    fixed['metadata']['source_file'] = args.input_json
    fixed['metadata']['umbilicus_file'] = args.umbilicus
    fixed['metadata']['fix_description'] = (
        'Each point collection was reordered independently. Z-oriented collections were '
        f'sorted by {args.z_direction} z using PCA z-axis threshold {args.z_axis_min}; '
        'all other collections were ordered by the diameter path of a minimum spanning '
        'tree in umbilicus-relative polar coordinates. Point ids were reassigned to encode '
        'the fixed order in the JSON; creation_time was also rewritten unless '
        '--preserve-creation-time was used.'
    )
    fixed['metadata']['fix_stats'] = stats

    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f'.{Path(output_path).stem}.',
        suffix=Path(output_path).suffix or '.json',
        dir=output_dir,
        text=True,
    )
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(fixed, f, indent=2)
            f.write('\n')
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise

    print(f'wrote {output_path}')
    print(f"collections: {stats['collections']}")
    print(f"points: {stats['points']}")
    print(f"z-oriented collections: {stats['z_oriented_collections']}")
    print(f"MST collections: {stats['mst_collections']}")


if __name__ == '__main__':
    main()
