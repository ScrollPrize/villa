import os
from pathlib import Path
from typing import List, Dict, Union, Tuple
import zarr
import tifffile
import numpy as np
import torch
from torch.utils.data import Dataset
from helpers import generate_positions
from pytorch3dunet.augment.transforms import Standardize

class InferenceDataset(Dataset):
    def __init__(
            self,
            input_path: str,
            targets: List[Dict],
            patch_size=(128, 128, 128),
            input_format: str = 'zarr',
            overlap: float = 0.25,
            load_all: bool = False
            ):

        self.input_path = input_path
        self.input_format = input_format
        self.targets = targets
        self.patch_size = patch_size
        self.normalization = Standardize(channelwise=False)
        self.overlap = overlap
        self.load_all = load_all

        if input_format == 'zarr':
            self.input_array = zarr.open(self.input_path, mode='r')

        if load_all:
            self.input_array = self.input_array[:]

        self.input_shape = self.input_array.shape
        self.input_dtype = self.input_array.dtype

        pZ, pY, pX = patch_size
        minz, maxz = 0, self.input_shape[0]
        miny, maxy = 0, self.input_shape[1]
        minx, maxx = 0, self.input_shape[2]

        # generate all coordinates
        z_step = int(round(pZ * (1 - self.overlap)))
        y_step = int(round(pY * (1 - self.overlap)))
        x_step = int(round(pX * (1 - self.overlap)))

        z_positions = generate_positions(minz, maxz, pZ, z_step)
        y_positions = generate_positions(miny, maxy, pY, y_step)
        x_positions = generate_positions(minx, maxx, pX, x_step)

        self.all_positions = []
        for z in z_positions:
            for y in y_positions:
                for x in x_positions:
                    self.all_positions.append((z, y, x))


    def __len__(self):
        return len(self.all_positions)

    def __getitem__(self, idx):
        z, y, x = self.all_positions[idx]
        patch = self.input_array[z:z+self.patch_size[0], y:y+self.patch_size[1], x:x+self.patch_size[2]]

        if self.input_dtype == np.uint8:
            patch = patch.astype(np.float32) / 255.0
        elif self.input_dtype == np.uint16:
            patch = patch.astype(np.float32) / 65535.0

        patch = self.normalization(patch)
        patch = patch.astype(np.float32)
        patch = patch[np.newaxis, ...]
        patch = torch.from_numpy(patch)
        return {"image": patch, "index":idx}

