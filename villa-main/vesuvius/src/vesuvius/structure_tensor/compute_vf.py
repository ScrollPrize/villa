#compute_vf.py
import argparse
import torch
import zarr
from numcodecs import Blosc
try:
    # When executed as a module: python -m structure_tensor.compute_vf
    from .create_vf import VectorFieldComputer
except ImportError:  # Fallback when run as a loose script
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
    p.add_argument('--ome-u8', action='store_true',
                   help="Write OME-ish uint8 layout (group/{z,y,x}/scale).")
    p.add_argument('--ome-only', action='store_true',
                   help="Write only the OME-ish layout, skip float U/V/N.")
    p.add_argument('--group-name', default='horizontal',
                   help="Top-level semantic group name (default: horizontal).")
    p.add_argument('--scale', default='0',
                   help="Scale name (string) for OME-ish datasets (default: 0).")
    p.add_argument('--downsample', type=int, default=1,
                   help="Integer downsample factor for OME-ish output (pick every Nth voxel).")
    p.add_argument('--write-confidence', action='store_true',
                   help="Also write a scalar confidence group in uint8.")
    p.add_argument('--export-field', default='V', choices=['U','V','N','u','v','n'],
                   help="Which vector to export to OME-ish layout (default: V).")
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

    compressor = Blosc(
        cname=args.cname,
        clevel=args.clevel,
        shuffle=Blosc.SHUFFLE
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
        compressor=compressor,
        ome_u8=args.ome_u8 or args.ome_only,
        group_name=args.group_name,
        output_downsample=args.downsample,
        output_ome_scale=args.scale,
        write_confidence=args.write_confidence,
        ome_only=args.ome_only,
        export_field=args.export_field,
    )

if __name__ == '__main__':
    main()
