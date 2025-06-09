# vector_fields.py

import torch
import zarr
import numpy as np
from data.utils import open_zarr
from typing import Dict, Tuple, Optional
from tqdm.auto import tqdm

from smooth_vf import VectorFieldModule
from vf_io import load_density_mask, load_eigenvector_field

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
    ):
        # open input to get shape
        inp = open_zarr(self.input_zarr, mode='r')
        Z, Y, X = inp.shape[-3:]
        cz, cy, cx = chunk_size

        # open/create output
        root = zarr.open_group(output_zarr, mode='a')
        U_ds = root.require_dataset('U',   shape=(3,Z,Y,X), chunks=(3,cz,cy,cx),
                                    dtype=np.float32, compressor=compressor, overwrite=True)
        V_ds = root.require_dataset('V',   shape=(3,Z,Y,X), chunks=(3,cz,cy,cx),
                                    dtype=np.float32, compressor=compressor, overwrite=True)
        N_ds = root.require_dataset('N',   shape=(3,Z,Y,X), chunks=(3,cz,cy,cx),
                                    dtype=np.float32, compressor=compressor, overwrite=True)
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
                # 4) crop & accumulate
                iz0, iy0, ix0 = z0-zp, y0-yp, x0-xp
                U_block += U_ext[:, iz0:iz0+Dz, iy0:iy0+Dy, ix0:ix0+Dx]
                V_block += V_ext[:, iz0:iz0+Dz, iy0:iy0+Dy, ix0:ix0+Dx]
                N_block += N_ext[:, iz0:iz0+Dz, iy0:iy0+Dy, ix0:ix0+Dx]

            # 5) write out
            U_ds[:, z0:z1, y0:y1, x0:x1] = U_block.cpu().numpy()
            V_ds[:, z0:z1, y0:y1, x0:x1] = V_block.cpu().numpy()
            N_ds[:, z0:z1, y0:y1, x0:x1] = N_block.cpu().numpy()
            torch.cuda.empty_cache()

        print(f"✔ chunked U, V written to {output_zarr}")
