#optimize_fields/dataset.py

from torch.utils.data import Dataset
import zarr
import numpy as np
from typing import List, Tuple

# optimize_fields/dataset.py

class ChunkDataset(Dataset):
    def __init__(
        self,
        zarr_U, zarr_V, zarr_N,
        chunk_size: Tuple[int,int,int],
        ghost: int = 1             # number of voxels to pad on each side
    ):

        self.U      = zarr_U
        self.V      = zarr_V
        self.N      = zarr_N
        self.ghost  = ghost

        self.cz, self.cy, self.cx = chunk_size
        _, self.Z, self.Y, self.X = self.U.shape
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
        U  = self.U     [:, zp0:zp1, yp0:yp1, xp0:xp1].astype('float32')
        V  = self.V     [:, zp0:zp1, yp0:yp1, xp0:xp1].astype('float32')
        N  = self.N     [:, zp0:zp1, yp0:yp1, xp0:xp1].astype('float32')

        # return both the original and padded bounds
        return (
            (z0,z1,y0,y1,x0,x1),
            (zp0,zp1,yp0,yp1,xp0,xp1),
            U, V, N
        )

