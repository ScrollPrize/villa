#!/usr/bin/env python3
"""Extract surface tracks from a surface-prediction volume into a tracks DBM.

The predictions are binarized (``> 0``) and cut into thin slabs: horizontal
ribbons ``z_chunk_depth_h`` voxels deep every ``z_chunk_stride_h`` voxels, then
vertical zx and zy slabs ``yx_chunk_thickness_v`` voxels thick every
``yx_stride_v`` voxels across the occupied yx range. In every slab the
connected components are labelled at full resolution, max-pooled by
``downsample_factor``, filtered by size and skeletonized, and, with the default
``path_mode`` ``'interjoint'``, each chain of the skeletons between branch or
end points with at least 10 vertices becomes one track.

The DBM (stdlib ``dbm``) maps ``h:{z}``, ``vy:{y}`` and ``vx:{x}`` to a pickled
list of ``(N, 3)`` int32 ZYX arrays in full-resolution voxels. Keys already in
the DBM are skipped, so an interrupted run resumes where it stopped; delete the
DBM to recompute it. With --packed-store (default:
``write_native_packed_store``), the adjacent packed store read by
``fit_spiral.py`` is written at the end.

Two backends write the same keys in the same format:

  gpu  Brook (brook-cu12) and cuCIM on NVIDIA GPUs of compute capability 8.0 or
       newer, Linux x86-64 only, one process per GPU (``surface_tracks_gpu.py``).
  cpu  Kimimaro on the CPU, one slab at a time.

Brook implements Kimimaro's algorithm on the GPU; its skeletons are close to
Kimimaro's but not guaranteed to be identical. ``--backend auto`` uses the GPU
backend when a compatible GPU is present and ``path_mode`` is ``'interjoint'``,
and Kimimaro otherwise.

Example:

    python spiral-fitting/extract_surface_tracks.py \\
        --predictions /path/to/<surface-predictions>.zarr/0 \\
        --out /path/to/<dataset>/tracks/<name>.dbm \\
        --z-min 10900 --z-max 11300 --gpus 0,1

Options default to the configuration at the top of this file.
"""

from __future__ import annotations

import argparse
import dbm
import importlib.util
import os
import pickle
import re
import sys

import numpy as np


predictions_zarr_path = 's3://<bucket>/<volpkg>/volumes/<surface-predictions>.zarr/0'
tracks_dbm_path = '/path/to/<dataset>/tracks/<name>.dbm'
write_native_packed_store = True
skeleton_backend = 'auto'  # 'gpu' = Brook, 'cpu' = Kimimaro, 'auto' = gpu when it can run, else cpu

# All wrt the original (full-resolution) volume; downsampling is applied internally after CC extraction
downsample_factor = 4
z_min, z_max = 10900, 11300
z_chunk_depth_h = 4  # thickness of horizontal ribbons
z_chunk_stride_h = 16  # stride between successive ribbons
yx_chunk_thickness_v = 4  # thickness of vertical slabs
yx_stride_v = 64  # stride between successive vertical slabs
dust_threshold = 400  # remove components smaller than this many voxels
max_area_threshold = 640_000  # drop components larger than this many voxels
path_mode = 'interjoint'  # 'interjoint' = every chain between branch/terminal nodes; 'maximal_chain' = greedy longest-path peel

BACKENDS = ('auto', 'gpu', 'cpu')
# An unfilled placeholder such as '<bucket>' in a configured path.
_PLACEHOLDER = re.compile(r'<[^<>/]+>')


def open_predictions(path):
    """Open the predictions array read-only with vesuvius's ``open_zarr`` (local path or URL)."""
    vesuvius_src = f'{os.path.dirname(os.path.abspath(__file__))}/../vesuvius/src'
    if vesuvius_src not in sys.path:
        sys.path.insert(0, vesuvius_src)
    from vesuvius.data.utils import open_zarr
    return open_zarr(path, mode='r')


def downsample_maxpool(x, factor):
    if factor == 1:
        return x
    cropped = tuple(s - s % factor for s in x.shape)
    x = x[tuple(slice(0, s) for s in cropped)]
    reshape = []
    for s in x.shape:
        reshape += [s // factor, factor]
    x = x.reshape(reshape)
    for axis in range(len(reshape) - 1, 0, -2):
        x = np.max(x, axis=axis)
    return x


def extract_inter_branch_paths(graph):
    # Every maximal degree-2 chain between critical (non-degree-2) nodes.
    paths = []
    visited_edges = set()
    critical_nodes = [n for n in graph.nodes() if graph.degree(n) != 2]
    for node in critical_nodes:
        for neighbor in graph.neighbors(node):
            edge = tuple(sorted([node, neighbor]))
            if edge in visited_edges:
                continue
            path = [node, neighbor]
            visited_edges.add(edge)
            current = neighbor
            while graph.degree(current) == 2:
                next_nodes = [n for n in graph.neighbors(current) if n != path[-2]]
                if not next_nodes:
                    break
                next_node = next_nodes[0]
                edge = tuple(sorted([current, next_node]))
                if edge in visited_edges:
                    break
                path.append(next_node)
                visited_edges.add(edge)
                current = next_node
            paths.append(path)
    return paths


def skeletonize_kwargs():
    """Skeletonization options that determine the result, shared by both backends."""
    return dict(
        teasar_params={'scale': 1., 'const': 2.},
        anisotropy=(downsample_factor, downsample_factor, downsample_factor),  # skeletons therefore have not-downsampled coordinates
        dust_threshold=dust_threshold,
        fix_branching=True,
        fix_borders=True,
        fill_holes=False,
    )


def geometry():
    """Slab layout and component-size filter, in full-resolution voxels, for the GPU backend."""
    return dict(
        downsample_factor=downsample_factor,
        z_chunk_depth_h=z_chunk_depth_h,
        z_chunk_stride_h=z_chunk_stride_h,
        yx_chunk_thickness_v=yx_chunk_thickness_v,
        yx_stride_v=yx_stride_v,
        dust_threshold=dust_threshold,
        max_area_threshold=max_area_threshold,
    )


def get_skeleton_tracks(cc_labels, offset_zyx, verbose=False, skeletonize=None):
    """Tracks of one slab; ``skeletonize`` defaults to ``kimimaro.skeletonize``.

    Any function with Kimimaro's signature can be passed, e.g. ``brook.skeletonize``.
    """
    import networkx as nx
    if skeletonize is None:
        import kimimaro
        skeletonize = kimimaro.skeletonize
    skeletons = skeletonize(
        cc_labels,
        **skeletonize_kwargs(),
        progress=verbose,
        parallel=8,
        parallel_chunk_size=250,
        in_place=True,
    )
    offset = np.asarray(offset_zyx, dtype=np.int64)
    tracks = []
    for skeleton in skeletons.values():
        if path_mode == 'interjoint':
            graph = nx.Graph()
            graph.add_edges_from(skeleton.edges)
            for path_vertex_indices in extract_inter_branch_paths(graph):
                if len(path_vertex_indices) < 10:
                    continue
                coords = skeleton.vertices[path_vertex_indices].astype(np.int64)
                tracks.append((coords + offset).astype(np.int32))
        elif path_mode == 'maximal_chain':
            while True:
                if len(skeleton.edges) == 0:
                    break
                paths = skeleton.interjoint_paths()
                longest_path_vertex_zyxs = max(paths, key=len)
                if len(longest_path_vertex_zyxs) < 10:
                    break
                coords = longest_path_vertex_zyxs.astype(np.int64)
                tracks.append((coords + offset).astype(np.int32))
                longest_path_vertex_indices = set(np.where((longest_path_vertex_zyxs[:, None, :] == skeleton.vertices[None, :, :]).all(axis=-1))[1])
                skeleton.edges = np.asarray(
                    [edge for edge in skeleton.edges if edge[0] not in longest_path_vertex_indices and edge[1] not in longest_path_vertex_indices],
                    dtype=np.uint32,
                )
        else:
            assert False, f'unknown path_mode {path_mode!r}'
    return tracks


def prepare_cc_labels(predictions_binary):
    # CC at native resolution, then max-pool down, then dust by combined min/max area.
    # Area thresholds are specified in full-res voxels; convert to downsampled by ds**3.
    import cc3d
    cc_labels, _ = cc3d.connected_components(predictions_binary, connectivity=6, return_N=True)
    cc_labels = downsample_maxpool(cc_labels, downsample_factor)
    scale = downsample_factor ** 3
    cc3d.dust(
        cc_labels,
        threshold=[max(1, dust_threshold // scale), max(1, max_area_threshold // scale)],
        in_place=True,
        precomputed_ccl=True,
    )
    return cc_labels


def extract_horizontal(predictions_zarr_array, tracks_db):
    from tqdm import tqdm
    z_limit = predictions_zarr_array.shape[0]
    for z_chunk_min in tqdm(list(range(z_min, z_max, z_chunk_stride_h)), desc='horizontal ribbons'):
        db_key = f'h:{z_chunk_min}'
        if db_key in tracks_db:
            continue
        z_chunk_max = min(z_chunk_min + z_chunk_depth_h, z_max, z_limit)
        if z_chunk_max <= z_chunk_min:
            continue
        predictions = predictions_zarr_array[z_chunk_min : z_chunk_max]
        predictions = (predictions > 0).astype(np.uint8)
        if predictions.max() == 0:
            tracks_db[db_key] = pickle.dumps([])
            continue
        cc_labels = prepare_cc_labels(predictions)
        tracks = get_skeleton_tracks(cc_labels, offset_zyx=(z_chunk_min, 0, 0))
        tracks_db[db_key] = pickle.dumps(tracks)


def find_yx_range(predictions_zarr_array):
    # Returns full-resolution yx bounds.
    from tqdm import tqdm
    shape = predictions_zarr_array.shape
    z_limit = shape[0]
    z_range_min = min(z_min, z_limit)
    z_range_max = min(z_max, z_limit)
    min_yx = np.array([shape[1], shape[2]])
    max_yx = np.array([0, 0])
    step = max(1, (z_range_max - z_range_min) // 20)
    for z in tqdm(range(z_range_min, z_range_max, step), desc='finding yx range'):
        predictions = predictions_zarr_array[z]
        yxs = np.stack(np.where(predictions > 0), axis=-1)
        if len(yxs) > 0:
            min_yx = np.minimum(min_yx, yxs.min(axis=0))
            max_yx = np.maximum(max_yx, yxs.max(axis=0))
    return min_yx, max_yx


def extract_vertical(predictions_zarr_array, tracks_db, axis, min_yx, max_yx):
    # axis='y': iterate over y, slicing zx slabs. axis='x': iterate over x, slicing zy slabs.
    # min_yx / max_yx / yx_stride_v and w are all in full-resolution units; the slab thickness
    # in full-res voxels is yx_chunk_thickness_v * downsample_factor.
    from tqdm import tqdm
    if axis == 'y':
        lo, hi = min_yx[0], max_yx[0]
        axis_idx = 1
        key_prefix = 'vy'
    elif axis == 'x':
        lo, hi = min_yx[1], max_yx[1]
        axis_idx = 2
        key_prefix = 'vx'
    else:
        assert False

    z_limit = predictions_zarr_array.shape[0]
    z_range_min = min(z_min, z_limit)
    z_range_max = min(z_max, z_limit)
    if z_range_max <= z_range_min:
        return
    shape_along = predictions_zarr_array.shape[axis_idx]

    for w in tqdm(list(range(lo, hi, yx_stride_v)), desc=f'vertical {axis}-stride slabs'):
        db_key = f'{key_prefix}:{w}'
        if db_key in tracks_db:
            continue
        w_max = min(w + yx_chunk_thickness_v, shape_along)
        if w_max - w < downsample_factor:
            tracks_db[db_key] = pickle.dumps([])
            continue
        if axis == 'y':
            predictions = predictions_zarr_array[z_range_min:z_range_max, w:w_max, :]  # (z, t, x)
            offset_zyx = (z_range_min, w, 0)
        else:
            predictions = predictions_zarr_array[z_range_min:z_range_max, :, w:w_max]  # (z, y, t)
            offset_zyx = (z_range_min, 0, w)
        predictions = (predictions > 0).astype(np.uint8)
        if predictions.max() == 0:
            tracks_db[db_key] = pickle.dumps([])
            continue
        cc_labels = prepare_cc_labels(predictions)
        tracks = get_skeleton_tracks(cc_labels, offset_zyx=offset_zyx)
        tracks_db[db_key] = pickle.dumps(tracks)


def run_cpu(predictions_path, tracks_dbm_path):
    """Extract every slab in turn on the CPU (Kimimaro), writing each key as it completes."""
    print(f'opening {predictions_path}')
    predictions_zarr_array = open_predictions(predictions_path)

    with dbm.open(tracks_dbm_path, 'c') as tracks_db:

        print('extracting horizontal ribbons')
        extract_horizontal(predictions_zarr_array, tracks_db)

        print('finding yx range for vertical passes')
        min_yx, max_yx = find_yx_range(predictions_zarr_array)
        print(f'  yx range (full-res): {min_yx} .. {max_yx}')

        print('extracting vertical zx-plane tracks')
        extract_vertical(predictions_zarr_array, tracks_db, axis='y', min_yx=min_yx, max_yx=max_yx)

        print('extracting vertical zy-plane tracks')
        extract_vertical(predictions_zarr_array, tracks_db, axis='x', min_yx=min_yx, max_yx=max_yx)


def run_gpu(predictions_path, tracks_dbm_path, gpus=None):
    """Extract the same slabs with Brook on the selected GPUs (``surface_tracks_gpu.py``)."""
    import surface_tracks_gpu
    return surface_tracks_gpu.run(
        predictions_path, tracks_dbm_path, z_min=z_min, z_max=z_max, geometry=geometry(),
        skeletonize_kwargs=skeletonize_kwargs(), open_array=open_predictions, gpus=gpus)


def _have_kimimaro():
    return importlib.util.find_spec('kimimaro') is not None


def _kimimaro_missing(context):
    # On Linux x86-64 Kimimaro is an optional extra, because a GPU install does not need it.
    return SystemExit(
        f'{context}, and Kimimaro (the CPU backend) is not installed. On Linux x86-64 it is '
        'optional: run `uv sync --extra cpu` in spiral-fitting/ to add it.')


def select_backend(requested, gpus=None):
    """Return ``'gpu'`` or ``'cpu'`` for ``requested`` (``'auto'``, ``'gpu'`` or ``'cpu'``).

    Prints the backend and the reason; raises ``SystemExit`` when the requested
    backend cannot run. With ``gpus`` given, ``'auto'`` does not fall back to the CPU.
    """
    if requested not in BACKENDS:
        raise SystemExit(f'unknown backend {requested!r}; expected one of {", ".join(BACKENDS)}')
    if requested == 'cpu':
        if not _have_kimimaro():
            raise _kimimaro_missing('--backend cpu was requested')
        print('backend: cpu (Kimimaro), as requested')
        return 'cpu'

    # The GPU backend turns skeletons into tracks with fast_tracks, which implements
    # the 'interjoint' walk only.
    if path_mode != 'interjoint':
        ok = False
        reason = f"the GPU backend supports path_mode 'interjoint' only, not {path_mode!r}"
    else:
        try:
            import surface_tracks_gpu
        except ImportError as exc:
            ok, reason = False, f'surface_tracks_gpu cannot be imported ({exc})'
        else:
            ok, reason = surface_tracks_gpu.check_available(gpus)

    if requested == 'gpu':
        if not ok:
            raise SystemExit(
                f'--backend gpu was requested, but the GPU backend cannot run: {reason}')
        print(f'backend: gpu (Brook), as requested: {reason}')
        return 'gpu'
    if ok:
        print(f'backend: gpu (Brook): {reason}')
        return 'gpu'
    if gpus is not None:
        raise SystemExit(f'--gpus was given, but the GPU backend cannot run: {reason}')
    if not _have_kimimaro():
        raise _kimimaro_missing(f'The GPU backend cannot run ({reason})')
    print(f'backend: cpu (Kimimaro), because {reason}')
    return 'cpu'


def write_packed_store(tracks_dbm_path):
    # The DBM remains the resumable extraction format and compatibility
    # source. Fits and crossing builds consume this adjacent packed store.
    from tracks import _packed_store_if_current, write_packed_track_store
    if _packed_store_if_current(tracks_dbm_path) is None:
        write_packed_track_store(
            tracks_dbm_path, force=True, show_progress=True)


def parse_args(argv=None):
    from runners.run_single import parse_gpu_ids

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--predictions', metavar='URL_OR_PATH', default=predictions_zarr_path,
        help='surface-prediction zarr array, a local path or a URL (default: %(default)s)',
    )
    parser.add_argument(
        '--out', metavar='DBM', default=tracks_dbm_path,
        help='tracks DBM to create or resume (default: %(default)s)',
    )
    parser.add_argument(
        '--z-min', metavar='Z', type=int, default=z_min,
        help='first z voxel of the extraction (default: %(default)s)',
    )
    parser.add_argument(
        '--z-max', metavar='Z', type=int, default=z_max,
        help='z voxel where the extraction ends, exclusive (default: %(default)s)',
    )
    parser.add_argument(
        '--backend', choices=BACKENDS, default=skeleton_backend,
        help='Brook on NVIDIA GPUs, Kimimaro on the CPU, or auto: the GPU backend '
             'when it can run, else Kimimaro (default: %(default)s)',
    )
    parser.add_argument(
        '--gpus', metavar='DEVICE[,DEVICE...]', type=parse_gpu_ids,
        help='GPUs for the GPU backend, as nvidia-smi indices; with --backend auto, --gpus '
             'makes the GPU backend required: an error instead of the fall-back to Kimimaro '
             '(default: all visible GPUs)',
    )
    parser.add_argument(
        '--packed-store', action=argparse.BooleanOptionalAction,
        default=write_native_packed_store,
        help='write the packed track store next to the DBM when done; it needs PyTorch '
             '(default: %(default)s)',
    )
    args = parser.parse_args(argv)
    for option, name, value in (
            ('--predictions', 'predictions_zarr_path', args.predictions),
            ('--out', 'tracks_dbm_path', args.out)):
        if _PLACEHOLDER.search(value):
            parser.error(
                f'{option} {value!r} still contains a placeholder (<...>): pass a real '
                f'{option} or set {name} at the top of extract_surface_tracks.py')
    if args.z_min < 0 or args.z_min >= args.z_max:
        parser.error('--z-min must be non-negative and less than --z-max')
    if args.backend == 'cpu' and args.gpus is not None:
        parser.error('--gpus applies to the GPU backend only, not to --backend cpu')
    return args


def main(argv=None):
    # The slab functions above read the z range from the module configuration.
    global z_min, z_max
    args = parse_args(argv)
    z_min, z_max = args.z_min, args.z_max

    assert z_chunk_depth_h >= downsample_factor
    assert yx_chunk_thickness_v >= downsample_factor

    backend = select_backend(args.backend, gpus=args.gpus)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if backend == 'gpu':
        run_gpu(args.predictions, args.out, gpus=args.gpus)
    else:
        run_cpu(args.predictions, args.out)

    if args.packed_store:
        write_packed_store(args.out)


if __name__ == '__main__':
    np.random.seed(0)
    main()
