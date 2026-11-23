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
  cpu  Kimimaro on the CPU, one slab at a time, or several with
       --slab-workers; vc_reader labels connected components on the CPU.

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
slab_workers = 1  # CPU backend: slabs processed at once by worker processes
# CPU backend: Kimimaro processes per slab. Kimimaro spawns a new pool of them for every slab,
# which on Windows and macOS takes far longer than the work they share out; use --slab-workers
# there to run slabs in parallel instead.
kimimaro_parallel = 8 if sys.platform.startswith('linux') else 1

BACKENDS = ('auto', 'gpu', 'cpu')
# An unfilled placeholder such as '<bucket>' in a configured path.
_PLACEHOLDER = re.compile(r'<[^<>/]+>')


def open_predictions(path):
    """Open the predictions array read-only with zarr: a local path, an http(s) URL, or an
    s3:// URL read with the environment's AWS credentials."""
    import zarr
    if path.startswith('s3://'):
        return zarr.open_array(path, mode='r', storage_options={'anon': False})
    if path.startswith(('http://', 'https://')):
        return zarr.open_array(path, mode='r', storage_options={})
    return zarr.open_array(path, mode='r')


def downsample_maxpool(x, factor):
    """Max over ``factor``-sized cells, cropping each axis to a multiple of ``factor``."""
    if factor == 1:
        return x
    x = x[tuple(slice(0, s - s % factor) for s in x.shape)]
    # one axis at a time, as an element-wise maximum of strided views: a max over a length-factor
    # inner axis of a reshaped array is several times slower, and each pass shrinks the next
    for axis in range(x.ndim):
        def phase(i):
            return x[(slice(None),) * axis + (slice(i, None, factor),)]
        out = np.maximum(phase(0), phase(1))
        for i in range(2, factor):
            np.maximum(out, phase(i), out=out)
        x = out
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
        # one below prepare_cc_labels' downsampled cutoff (it keeps counts >= it); Kimimaro and Brook keep counts > this
        dust_threshold=max(1, dust_threshold // downsample_factor ** 3) - 1,
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
    kwargs = skeletonize_kwargs()
    if skeletonize is None:
        import kimimaro
        skeletonize = kimimaro.skeletonize
        if sorted(cc_labels.shape)[1] == 1:
            kwargs['fix_borders'] = False  # Kimimaro's border targets index out of bounds on a one-voxel face
    skeletons = skeletonize(
        cc_labels,
        **kwargs,
        progress=verbose,
        parallel=kimimaro_parallel,
        parallel_chunk_size=250,
        in_place=True,
    )
    offset = np.asarray(offset_zyx, dtype=np.int64)
    if path_mode == 'interjoint':
        # fast_tracks walks every skeleton as extract_inter_branch_paths does on its networkx graph,
        # without building one (compiled with numba where it is installed): the same tracks, in
        # the same order
        import fast_tracks
        return fast_tracks.tracks_from_skeletons(skeletons.values(), offset)
    assert path_mode == 'maximal_chain', f'unknown path_mode {path_mode!r}'
    tracks = []
    for skeleton in skeletons.values():
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
    return tracks


def prepare_cc_labels(predictions_binary):
    # CC at native resolution, then max-pool down, then dust by combined min/max area.
    # Area thresholds are specified in full-res voxels; convert to downsampled by ds**3.
    import cc3d
    from vc_reader.geometry import label_components
    cc_labels, _ = label_components(predictions_binary, connectivity=6)
    # cc3d.dust and Kimimaro expect unsigned label arrays; labels are non-negative, so a view of
    # vc_reader's int32 labels saves a full-resolution copy
    if cc_labels.dtype == np.int32:
        cc_labels = cc_labels.view(np.uint32)
    else:
        cc_labels = cc_labels.astype(np.uint32, copy=False)
    cc_labels = downsample_maxpool(cc_labels, downsample_factor)
    scale = downsample_factor ** 3
    cc3d.dust(
        cc_labels,
        threshold=[max(1, dust_threshold // scale), max(1, max_area_threshold // scale)],
        in_place=True,
        precomputed_ccl=True,
    )
    return cc_labels


def _slab_tracks(predictions_zarr_array, index, offset_zyx):
    # Pickled tracks of one slab (read, binarize, CC, skeletonize), as stored in the DBM.
    predictions = binarize(predictions_zarr_array[index])
    if predictions.max() == 0:
        return pickle.dumps([])
    cc_labels = prepare_cc_labels(predictions)
    # the full-resolution slab is the largest array of a slab; free it before skeletonizing
    del predictions
    return pickle.dumps(get_skeleton_tracks(cc_labels, offset_zyx=offset_zyx))


def binarize(predictions):
    """``(predictions > 0)`` as uint8, in place for the uint8 arrays zarr returns."""
    if predictions.dtype == np.uint8 and predictions.flags.writeable and predictions.flags.owndata:
        return np.minimum(predictions, 1, out=predictions)
    return (predictions > 0).astype(np.uint8)


_worker_array = None  # the opened array, inherited by forked workers
# Module settings the slab workers read. Spawned workers re-import this file, so they get
# the parent's values (possibly set from the command line) through _worker_init.
_WORKER_SETTINGS = ('downsample_factor', 'dust_threshold', 'max_area_threshold', 'path_mode',
                    'kimimaro_parallel')


def _worker_init(array, settings):
    global _worker_array
    _worker_array = array
    globals().update(settings)


def _worker_slab(job):
    key, index, offset_zyx = job
    return key, _slab_tracks(_worker_array, index, offset_zyx)


def _run_slabs(predictions_zarr_array, tracks_db, jobs, desc):
    # Compute (key, index, offset) jobs not yet in the DBM, slab_workers slabs at a time.
    # Results are stored in job order, so an interrupted run resumes as before.
    from tqdm import tqdm
    jobs = [j for j in jobs if j[0] not in tracks_db]
    if not jobs:
        return
    if slab_workers <= 1:
        for key, index, offset in tqdm(jobs, desc=desc):
            tracks_db[key] = _slab_tracks(predictions_zarr_array, index, offset)
        return
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    global _worker_array
    if sys.platform.startswith('linux'):
        # Forked workers inherit the opened array and the module settings.
        _worker_array = predictions_zarr_array
        pool = ProcessPoolExecutor(slab_workers, mp_context=multiprocessing.get_context('fork'))
    else:
        # Windows and macOS: spawned workers receive the array (a zarr array pickles as its
        # path) and the settings once, when they start.
        settings = {k: globals()[k] for k in _WORKER_SETTINGS}
        pool = ProcessPoolExecutor(
            slab_workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=_worker_init, initargs=(predictions_zarr_array, settings))
    try:
        for key, data in tqdm(pool.map(_worker_slab, jobs), total=len(jobs), desc=desc):
            tracks_db[key] = data
    finally:
        # On an error or Ctrl-C, drop the slabs not started yet; running slabs finish first.
        pool.shutdown(cancel_futures=True)
        _worker_array = None


def horizontal_jobs(predictions_zarr_array, tracks_db):
    """(key, index, offset) jobs of the horizontal ribbons; ribbons too thin to have tracks are
    written as empty at once."""
    z_limit = predictions_zarr_array.shape[0]
    jobs = []
    for z_chunk_min in range(z_min, z_max, z_chunk_stride_h):
        z_chunk_max = min(z_chunk_min + z_chunk_depth_h, z_max, z_limit)
        if z_chunk_max <= z_chunk_min:
            continue
        key = f'h:{z_chunk_min}'
        if z_chunk_max - z_chunk_min < downsample_factor:
            # The last ribbon, clipped at z_max or the volume end, is thinner than one max-pool
            # cell: downsample_maxpool would crop it to nothing, so it has no tracks.
            if key not in tracks_db:
                tracks_db[key] = pickle.dumps([])
            continue
        jobs.append((key, slice(z_chunk_min, z_chunk_max), (z_chunk_min, 0, 0)))
    return jobs


def extract_horizontal(predictions_zarr_array, tracks_db):
    _run_slabs(predictions_zarr_array, tracks_db, horizontal_jobs(predictions_zarr_array, tracks_db),
               'horizontal ribbons')


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


def vertical_jobs(predictions_zarr_array, tracks_db, axis, min_yx, max_yx):
    """(key, index, offset) jobs of the vertical slabs along ``axis``; slabs too thin to have
    tracks are written as empty at once."""
    # axis='y': iterate over y, slicing zx slabs. axis='x': iterate over x, slicing zy slabs.
    # min_yx / max_yx / yx_stride_v and w are all in full-resolution units; the slab thickness
    # in full-res voxels is yx_chunk_thickness_v * downsample_factor.
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
    # Start the slabs on the downsample grid: horizontal-ribbon tracks lie on it, and crossings
    # are exact voxel matches, so slabs starting off the grid share no voxels with them.
    lo -= lo % downsample_factor

    z_limit = predictions_zarr_array.shape[0]
    z_range_min = min(z_min, z_limit)
    z_range_max = min(z_max, z_limit)
    if z_range_max <= z_range_min:
        return []
    shape_along = predictions_zarr_array.shape[axis_idx]

    jobs = []
    for w in range(lo, hi, yx_stride_v):
        db_key = f'{key_prefix}:{w}'
        if db_key in tracks_db:
            continue
        w_max = min(w + yx_chunk_thickness_v, shape_along)
        if w_max - w < downsample_factor or z_range_max - z_range_min < downsample_factor:
            tracks_db[db_key] = pickle.dumps([])
            continue
        if axis == 'y':
            index = (slice(z_range_min, z_range_max), slice(w, w_max), slice(None))  # (z, t, x)
            offset_zyx = (z_range_min, w, 0)
        else:
            index = (slice(z_range_min, z_range_max), slice(None), slice(w, w_max))  # (z, y, t)
            offset_zyx = (z_range_min, 0, w)
        jobs.append((db_key, index, offset_zyx))
    return jobs


def extract_vertical(predictions_zarr_array, tracks_db, axis, min_yx, max_yx):
    jobs = vertical_jobs(predictions_zarr_array, tracks_db, axis, min_yx, max_yx)
    _run_slabs(predictions_zarr_array, tracks_db, jobs, f'vertical {axis}-stride slabs')


def check_vertical_slab_grid(tracks_dbm_path):
    """Refuse to resume a DBM with vertical slabs from before they started on the downsample
    grid: resuming it would add aligned slabs beside its old ones."""
    if not dbm.whichdb(tracks_dbm_path):
        return
    with dbm.open(tracks_dbm_path, 'w') as tracks_db:
        keys = [key.decode() for key in tracks_db.keys()]
    for prefix in ('vy', 'vx'):
        starts = sorted(int(key.split(':')[1]) for key in keys if key.startswith(f'{prefix}:'))
        if starts and (starts[0] % downsample_factor
                       or any((w - starts[0]) % yx_stride_v for w in starts)):
            raise SystemExit(
                f'{tracks_dbm_path} has {prefix} slabs off the {downsample_factor}-voxel grid, so '
                'they cannot cross the horizontal tracks. Delete the DBM, extract it again, then '
                'rebuild its crossings with build_track_crossings.py --force.')


def run_cpu(predictions_path, tracks_dbm_path):
    """Extract every slab on the CPU (Kimimaro), writing each key as it completes."""
    check_vertical_slab_grid(tracks_dbm_path)
    print(f'opening {predictions_path}')
    predictions_zarr_array = open_predictions(predictions_path)

    with dbm.open(tracks_dbm_path, 'c') as tracks_db:

        print('finding yx range for vertical passes')
        min_yx, max_yx = find_yx_range(predictions_zarr_array)
        print(f'  yx range (full-res): {min_yx} .. {max_yx}')

        # All three passes go through one pool, so workers start once and no pass waits for
        # the last slabs of the one before it. Keys are still written in pass order.
        jobs = (horizontal_jobs(predictions_zarr_array, tracks_db)
                + vertical_jobs(predictions_zarr_array, tracks_db, 'y', min_yx, max_yx)
                + vertical_jobs(predictions_zarr_array, tracks_db, 'x', min_yx, max_yx))
        print('extracting horizontal ribbons, then zx- and zy-plane slabs')
        _run_slabs(predictions_zarr_array, tracks_db, jobs, 'slabs')


def run_gpu(predictions_path, tracks_dbm_path, gpus=None):
    """Extract the same slabs with Brook on the selected GPUs (``surface_tracks_gpu.py``)."""
    check_vertical_slab_grid(tracks_dbm_path)
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


def check_downsample_factor(tracks_dbm_path, factor):
    record = tracks_dbm_path + '.downsample_factor'
    with dbm.open(tracks_dbm_path, 'c') as tracks_db:
        resuming = bool(tracks_db.keys())
    if resuming:
        try:
            with open(record) as f:
                written = int(f.read())
        except (OSError, ValueError):
            raise SystemExit(f'{tracks_dbm_path} holds slabs, but {record} does not give the '
                             f'--downsample-factor they were extracted with; write that factor '
                             f'to {record} to resume, or pass a new --out') from None
        if written != factor:
            raise SystemExit(f'{tracks_dbm_path} holds slabs extracted with --downsample-factor '
                             f'{written}; resume it with that factor or pass a new --out')
        return
    with open(record + '.tmp', 'w') as f:
        f.write(f'{factor}\n')
    os.replace(record + '.tmp', record)


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
        '--downsample-factor', metavar='N', type=int, default=downsample_factor,
        help='max-pool factor applied to the labelled components before skeletonizing; '
             'use 1 when neighbouring sheets are only a few voxels apart. An existing DBM '
             'resumes only with the factor it was written with (default: %(default)s)',
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
        '--slab-workers', metavar='N', type=int, default=slab_workers,
        help='CPU backend: slabs processed at once in worker processes, each holding its own '
             'slab in memory; local predictions only (default: %(default)s)',
    )
    parser.add_argument(
        '--kimimaro-parallel', metavar='N', type=int, default=kimimaro_parallel,
        help='CPU backend: Kimimaro processes per slab; use 1 with --slab-workers > 1 '
             '(default: %(default)s)',
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
    # Otherwise horizontal and vertical tracks are sampled on mostly disjoint grids.
    ds = args.downsample_factor
    if (not 1 <= ds <= min(z_chunk_depth_h, yx_chunk_thickness_v)
            or z_chunk_stride_h % ds or yx_stride_v % ds):
        parser.error(f'--downsample-factor must be at most the slab thickness '
                     f'({min(z_chunk_depth_h, yx_chunk_thickness_v)}) and divide the slab '
                     f'strides ({z_chunk_stride_h} and {yx_stride_v})')
    if args.backend == 'cpu' and args.gpus is not None:
        parser.error('--gpus applies to the GPU backend only, not to --backend cpu')
    cpu_opts = ('slab_workers', 'kimimaro_parallel')
    if args.backend != 'cpu' and any(getattr(args, k) != parser.get_default(k) for k in cpu_opts):
        parser.error('--slab-workers and --kimimaro-parallel apply to --backend cpu only')
    if args.slab_workers < 1 or args.kimimaro_parallel < 1:
        parser.error('--slab-workers and --kimimaro-parallel must be at least 1')
    if args.slab_workers > 1 and '://' in args.predictions:
        parser.error('--slab-workers > 1 needs local predictions; workers cannot share a remote store')
    return args


def main(argv=None):
    # The slab functions above read the z range and downsample factor from the module configuration.
    global z_min, z_max, downsample_factor, slab_workers, kimimaro_parallel
    args = parse_args(argv)
    z_min, z_max = args.z_min, args.z_max
    downsample_factor = args.downsample_factor
    slab_workers, kimimaro_parallel = args.slab_workers, args.kimimaro_parallel

    backend = select_backend(args.backend, gpus=args.gpus)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    check_downsample_factor(args.out, args.downsample_factor)
    if backend == 'gpu':
        run_gpu(args.predictions, args.out, gpus=args.gpus)
    else:
        run_cpu(args.predictions, args.out)

    if args.packed_store:
        write_packed_store(args.out)


if __name__ == '__main__':
    np.random.seed(0)
    main()
