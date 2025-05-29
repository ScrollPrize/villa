#!/usr/bin/env python3
import argparse
import torch
import zarr
from create_vf import VectorFieldComputer

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute (u,v) vector fields from eigen‐analysis using VectorFieldComputer"
    )
    p.add_argument(
        '--input-zarr', required=True,
        help="Zarr volume with intensity / structure data"
    )
    p.add_argument(
        '--eigen', required=True, action='append',
        help="One or more eigen stores of form IDX:PATH, e.g. 0:vert.zarr 1:horiz.zarr"
    )
    p.add_argument(
        '--xi', type=float, default=5.0,
        help="Regularization strength ξ"
    )
    p.add_argument(
        '--device', default='cuda',
        help="Torch device (e.g. cpu or cuda:0)"
    )
    p.add_argument(
        '--output-zarr', required=True,
        help="Where to write the (u,v) fields .zarr"
    )
    p.add_argument(
        '--chunk-size', default="256,256,256",
        help="Patch size cz,cy,cx (comma‐separated)"
    )
    p.add_argument(
        '--cname', default='zstd',
        help="Blosc compressor name (default zstd)"
    )
    p.add_argument(
        '--clevel', type=int, default=3,
        help="Blosc compression level (default 3)"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # parse eigen→path map
    eigen_map = {}
    for e in args.eigen:
        idx_str, path = e.split(':', 1)
        eigen_map[int(idx_str)] = path

    # parse chunk size
    cz, cy, cx = map(int, args.chunk_size.split(','))

    compressor = zarr.Blosc(
        cname=args.cname,
        clevel=args.clevel,
        shuffle=zarr.Blosc.SHUFFLE
    )

    # set up device
    device = torch.device(args.device)

    # instantiate and run
    computer = VectorFieldComputer(
        input_zarr=args.input_zarr,
        eigen_zarrs=eigen_map,
        xi=args.xi,
        device=device
    )
    computer.compute_fields_zarr(
        output_zarr=args.output_zarr,
        chunk_size=(cz, cy, cx),
        compressor=compressor
    )

if __name__ == '__main__':
    main()
