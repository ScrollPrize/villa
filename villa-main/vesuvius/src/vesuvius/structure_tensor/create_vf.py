# create_vf.py
import torch
import zarr
import numpy as np
from vesuvius.data.utils import open_zarr
from typing import Dict, Tuple, Optional
from tqdm.auto import tqdm

try:
    # Module execution (recommended)
    from .smooth_vf import VectorFieldModule
    from .vf_io import load_density_mask, load_eigenvector_field
    from .vf_format import OMEU8VectorWriter
except ImportError:
    # Script execution fallback
    from smooth_vf import VectorFieldModule
    from vf_io import load_density_mask, load_eigenvector_field
    from vf_format import OMEU8VectorWriter

class VectorFieldComputer:
    def __init__(
        self,
        input_zarr: str,
        eigen_zarrs: Dict[int,str],
        xi: float,
        device: Optional[torch.device] = None,
    ):
        self.input_zarr = input_zarr
        self.eigen_zarrs = eigen_zarrs
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dtype = torch.float32

        # compiled smoothing module
        self.smoother = VectorFieldModule(xi=xi, device=self.device)

    def compute_fields_zarr(
        self,
        output_zarr: str,
        chunk_size: Tuple[int,int,int],
        compressor=None,
        *,
        ome_u8: bool = True,
        group_name: str = "horizontal",   # semantic name per Paul's spec
        output_downsample: int = 1,
        output_ome_scale: str = "0",
        write_confidence: bool = True,
        ome_only: bool = False,
        export_field: str = "V",          # which vector to export to OME-u8: "U" | "V" |
    ):
        # open input to get shape
        inp = open_zarr(self.input_zarr, mode='r')
        Z, Y, X = inp.shape[-3:]
        cz, cy, cx = chunk_size

        # prepare outputs
        root = zarr.open_group(output_zarr, mode='a')
        if not ome_only:
            U_ds = root.require_dataset('U', shape=(3,Z,Y,X), chunks=(3,cz,cy,cx),
                                        dtype=np.float32, compressor=compressor, overwrite=True)
            V_ds = root.require_dataset('V', shape=(3,Z,Y,X), chunks=(3,cz,cy,cx),
                                        dtype=np.float32, compressor=compressor, overwrite=True)
            N_ds = root.require_dataset('N', shape=(3,Z,Y,X), chunks=(3,cz,cy,cx),
                                        dtype=np.float32, compressor=compressor, overwrite=True)
        # OME-ish uint8 writer (writes component groups z/y/x at scale)
        writer = None
        if ome_u8:
            writer = OMEU8VectorWriter(
                output_path=output_zarr,
                group_name=group_name,
                vol_shape_zyx=(Z, Y, X),
                chunks_zyx=(cz, cy, cx),
                compressor=compressor,
                scale_name=output_ome_scale,
                downsample=output_downsample,
                make_confidence=write_confidence,
            )
        # generate chunk bounds
        def gen_bounds():
            for z0 in range(0, Z, cz):
                for y0 in range(0, Y, cy):
                    for x0 in range(0, X, cx):
                        z1, y1, x1 = min(z0+cz,Z), min(y0+cy,Y), min(x0+cx,X)
                        yield (z0,z1, y0,y1, x0,x1)
        bounds_list = list(gen_bounds())

        # outer progress bar over all chunks
        for bounds in tqdm(bounds_list, desc="Vector‐field chunks", unit="chunk"):
            z0,z1,y0,y1,x0,x1 = bounds
            Dz, Dy, Dx = z1-z0, y1-y0, x1-x0

            # halo
            rad = int(3 * self.smoother.xi)
            zp, zq = max(0, z0-rad), min(Z, z1+rad)
            yp, yq = max(0, y0-rad), min(Y, y1+rad)
            xp, xq = max(0, x0-rad), min(X, x1+rad)
            ext = (zp,zq, yp,yq, xp,xq)

            U_block = torch.zeros((3, Dz, Dy, Dx), device=self.device)
            V_block = torch.zeros_like(U_block)
            N_block = torch.zeros_like(U_block)
            # inner progress bar if you like per‐volume
            for s, e_path in self.eigen_zarrs.items():
                # 1) load masked eigenvectors
                mask = load_density_mask(self.input_zarr, s, ext).to(self.device)
                u_s  = load_eigenvector_field(e_path, 0, ext).to(self.device)
                v_s  = load_eigenvector_field(e_path, 1, ext).to(self.device)
                n_s = load_eigenvector_field(e_path, 2, ext).to(self.device)
                # 2) pack
                Su = (u_s * mask.unsqueeze(0)).unsqueeze(0)
                Sv = (v_s * mask.unsqueeze(0)).unsqueeze(0)
                Sn = (n_s * mask.unsqueeze(0)).unsqueeze(0)
                # 3) smooth & clone
                U_ext = self.smoother.smooth(Su).squeeze(0).clone()
                V_ext = self.smoother.smooth(Sv).squeeze(0).clone()
                N_ext = self.smoother.smooth(Sn).squeeze(0).clone()

                if torch.any(N_block != 0):
                    avg_N = torch.mean(N_ext, dim=(1,2,3))
                    avg_N_block = torch.mean(N_block, dim=(1,2,3))
                    if torch.sum(avg_N*avg_N_block) < 0:
                        N_ext *= -1
                        V_ext *= -1

                # 4) crop & accumulate
                iz0, iy0, ix0 = z0-zp, y0-yp, x0-xp
                U_block += U_ext[:, iz0:iz0+Dz, iy0:iy0+Dy, ix0:ix0+Dx]
                V_block += V_ext[:, iz0:iz0+Dz, iy0:iy0+Dy, ix0:ix0+Dx]
                N_block += N_ext[:, iz0:iz0+Dz, iy0:iy0+Dy, ix0:ix0+Dx]

            # 5) write out floats (unless ome-only)
            if not ome_only:
                U_ds[:, z0:z1, y0:y1, x0:x1] = U_block.cpu().numpy()
                V_ds[:, z0:z1, y0:y1, x0:x1] = V_block.cpu().numpy()
                N_ds[:, z0:z1, y0:y1, x0:x1] = N_block.cpu().numpy()

            # 6) write OME-ish uint8 for chosen vector (defaults to "horizontal" = V)
            if writer is not None:
                if export_field.upper() == "U":
                    H = U_block
                elif export_field.upper() == "N":
                    H = N_block
                else:
                    H = V_block
                # normalize to unit vectors; confidence = ||H|| clamped to [0,1]
                # (components are ordered z,y,x in channel dim)
                norm = torch.linalg.vector_norm(H, dim=0)  # [Dz,Dy,Dx]
                eps = 1e-8
                Hn = H / (norm + eps)
                # (3,Dz,Dy,Dx) -> (Dz,Dy,Dx,3)
                directions = torch.movedim(Hn, 0, -1).cpu().numpy()
                conf = None
                if write_confidence:
                    conf = torch.clamp(norm, 0.0, 1.0).cpu().numpy()
                writer.write_block(
                    z0=z0, z1=z1, y0=y0, y1=y1, x0=x0, x1=x1,
                    directions_block_zyx=directions,
                    confidence_block=conf,
                )
            torch.cuda.empty_cache()

        if writer is not None:
            writer.close()
        if not ome_only:
            print(f"✔ chunked U, V, N written to {output_zarr}")
        if writer is not None:
            print(f"✔ OME-ish uint8 written under '{group_name}/{{z,y,x}}/{output_ome_scale}' in {output_zarr}")
