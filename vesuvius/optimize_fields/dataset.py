#optimize_fields/dataset.py

from torch.utils.data import Dataset
import zarr
import numpy as np
from typing import List, Tuple

# optimize_fields/dataset.py

class ChunkDataset(Dataset):
    def __init__(
        self,
        zarr_u_init, zarr_v_init,
        zarr_U, zarr_V, zarr_n,
        chunk_size: Tuple[int,int,int],
        ghost: int = 1             # number of voxels to pad on each side
    ):
        self.u_init = zarr_u_init
        self.v_init = zarr_v_init
        self.U      = zarr_U
        self.V      = zarr_V
        self.n      = zarr_n
        self.ghost  = ghost

        self.cz, self.cy, self.cx = chunk_size
        _, self.Z, self.Y, self.X = self.u_init.shape
        self.bounds = self._make_bounds()

    def _make_bounds(self):
        b = []
        for z0 in range(0, self.Z, self.cz):
            z1 = min(z0 + self.cz, self.Z)
            for y0 in range(0, self.Y, self.cy):
                y1 = min(y0 + self.cy, self.Y)
                for x0 in range(0, self.X, self.cx):
                    x1 = min(x0 + self.cx, self.X)
                    b.append((z0,z1,y0,y1,x0,x1))
        return b

    def __len__(self):
        return len(self.bounds)

    def __getitem__(self, idx):
        z0,z1,y0,y1,x0,x1 = self.bounds[idx]
        g = self.ghost

        # padded bounds, clamped to volume
        zp0, zp1 = max(0, z0-g), min(self.Z, z1+g)
        yp0, yp1 = max(0, y0-g), min(self.Y, y1+g)
        xp0, xp1 = max(0, x0-g), min(self.X, x1+g)

        # load padded blocks
        u0 = self.u_init[:, zp0:zp1, yp0:yp1, xp0:xp1].astype('float32')
        v0 = self.v_init[:, zp0:zp1, yp0:yp1, xp0:xp1].astype('float32')
        U  = self.U     [:, zp0:zp1, yp0:yp1, xp0:xp1].astype('float32')
        V  = self.V     [:, zp0:zp1, yp0:yp1, xp0:xp1].astype('float32')
        n  = self.n     [:, zp0:zp1, yp0:yp1, xp0:xp1].astype('float32')

        # return both the original and padded bounds
        return (
            (z0,z1,y0,y1,x0,x1),
            (zp0,zp1,yp0,yp1,xp0,xp1),
            u0, v0, U, V, n
        )

