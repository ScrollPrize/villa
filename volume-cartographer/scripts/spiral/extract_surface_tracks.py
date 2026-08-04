
import os
import sys
import dbm
import pickle
import cc3d
import kimimaro
import numpy as np
import networkx as nx
from tqdm import tqdm

sys.path.insert(0, f'{os.path.dirname(__file__)}/../../../vesuvius/src')
from vesuvius.data.utils import open_zarr


predictions_zarr_path = 's3://volpkgs/s1_ds2.volpkg/volumes/2um_ds2_ps256_surf.zarr/0'
tracks_dbm_path = '/home/paul/projects/vesuvius-scrolls/spiral/tracks/2um_ds2_ps256_surf_v2.dbm'
write_native_packed_store = True

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


def get_skeleton_tracks(cc_labels, offset_zyx, verbose=False):
    skeletons = kimimaro.skeletonize(
        cc_labels,
        teasar_params={'scale': 1., 'const': 2.},
        anisotropy=(downsample_factor, downsample_factor, downsample_factor),  # skeletons therefore have not-downsampled coordinates
        dust_threshold=dust_threshold,
        fix_branching=True,
        fix_borders=True,
        fill_holes=False,
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


def main():

    assert z_chunk_depth_h >= downsample_factor
    assert yx_chunk_thickness_v >= downsample_factor

    print(f'opening {predictions_zarr_path}')
    predictions_zarr_array = open_zarr(predictions_zarr_path, mode='r')

    os.makedirs(os.path.dirname(tracks_dbm_path), exist_ok=True)
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

    if write_native_packed_store:
        # The DBM remains the resumable extraction format and compatibility
        # source. Fits and crossing builds consume this adjacent packed store.
        from tracks import _packed_store_if_current, write_packed_track_store
        if _packed_store_if_current(tracks_dbm_path) is None:
            write_packed_track_store(
                tracks_dbm_path, force=True, show_progress=True)


if __name__ == '__main__':
    np.random.seed(0)
    main()
