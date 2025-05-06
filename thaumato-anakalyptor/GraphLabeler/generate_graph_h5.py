#!/usr/bin/env python3
"""
Generate an HDF5 file containing node centroids and sample points
from a graph pickle file.

Usage:
    python generate_graph_h5.py input_graph.pkl output_nodes.h5 [--compression gzip|lzf]
"""
import argparse
import os
import sys
import pickle

try:
    import numpy as np
except ImportError:
    np = None
import h5py

import scroll_graph_util
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):
        return iterable

class SlimScrollGraph(scroll_graph_util.ScrollGraph):
    """
    Unpickled placeholder retaining only node centroids and sample_points.
    """
    def __setstate__(self, state):
        nodes_data = state.get('nodes', {}) or {}
        slim_nodes = {}
        total = len(nodes_data)
        for key, data in tqdm(nodes_data.items(), total=total, desc="Trimming nodes"):
            sp = data.get('sample_points')
            cent = data.get('centroid')
            sp_arr = np.asarray(sp, dtype=float) if sp is not None else None
            cent_arr = np.asarray(cent, dtype=float) if cent is not None else None
            slim_nodes[key] = {'sample_points': sp_arr, 'centroid': cent_arr}
        self.nodes = slim_nodes

class SlimUnpickler(pickle.Unpickler):
    """
    Redirect ScrollGraph and Graph classes to SlimScrollGraph.
    """
    def find_class(self, module, name):
        if module in ('scroll_graph_util', '__main__') and name in ('ScrollGraph', 'Graph'):
            return SlimScrollGraph
        return super().find_class(module, name)

def load_nodes_from_pkl(pkl_path):
    """
    Load pickled graph (full or slim) and return its nodes dict.
    Uses SlimUnpickler to downcast to SlimScrollGraph when needed.
    """
    with open(pkl_path, 'rb') as f:
        try:
            unp = SlimUnpickler(f)
            obj = unp.load()
        except Exception:
            f.seek(0)
            obj = pickle.load(f)
    # If already a dict of nodes
    if isinstance(obj, dict):
        return obj
    # If object has nodes attribute
    if hasattr(obj, 'nodes'):
        return obj.nodes
    raise ValueError(f"Unsupported pickled object type: {type(obj)}")

def node_key_to_group_name(key):
    """
    Convert a node key (tuple, list, array, or scalar) to a valid HDF5 group name.
    """
    # Sequence keys: join elements with underscore
    # For tuple or list keys, join elements with underscore
    if isinstance(key, (tuple, list)):
        try:
            parts = [str(int(x)) for x in key]
        except Exception:
            parts = [str(x) for x in key]
        return '_'.join(parts)
    # Fallback: use string representation
    # Scalar or other: use string representation
    return str(key)

def main():
    parser = argparse.ArgumentParser(
        description="Generate HDF5 of node centroids and sample points from graph pickle"
    )
    parser.add_argument('input_pkl', help='Path to graph pickle (.pkl)')
    parser.add_argument('output_h5', help='Output HDF5 file path (.h5)')
    parser.add_argument(
        '--compression', choices=['gzip', 'lzf'], default=None,
        help='Compression type for sample_points datasets'
    )
    args = parser.parse_args()
    # Prevent overwriting
    if os.path.exists(args.output_h5):
        print(f"Error: '{args.output_h5}' already exists. Aborting.", file=sys.stderr)
        sys.exit(1)
    # Load nodes dict
    try:
        nodes = load_nodes_from_pkl(args.input_pkl)
    except Exception as e:
        print(f"Failed to load nodes from '{args.input_pkl}': {e}", file=sys.stderr)
        sys.exit(1)
    # Write HDF5
    total_nodes = len(nodes)
    with h5py.File(args.output_h5, 'w') as h5f:
        for key, data in tqdm(nodes.items(), total=total_nodes, desc="Writing nodes"):
            grp_name = node_key_to_group_name(key)
            if grp_name in h5f:
                print(f"Warning: duplicate group name '{grp_name}', skipping.", file=sys.stderr)
                continue
            grp = h5f.create_group(grp_name)
            # Write centroid (node center) if available
            if data.get('centroid') is not None:
                grp.create_dataset('centroid', data=data['centroid'])
            elif data.get('position') is not None:
                grp.create_dataset('centroid', data=data['position'])
            # Write sample points (as float16) if available
            if data.get('sample_points') is not None:
                # Downcast sample points to float16 to save space
                sp = np.asarray(data['sample_points'], dtype=np.float16)
                grp.create_dataset(
                    'sample_points', data=sp,
                    compression=args.compression
                )
    print(f"HDF5 file '{args.output_h5}' written with {len(nodes)} nodes.")

if __name__ == '__main__':
    main()