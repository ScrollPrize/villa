from __future__ import annotations

import random

import numpy as np


class MaskingGenerator3d:
    def __init__(self, input_size: int | tuple[int, int, int]) -> None:
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size, input_size)
        self.depth, self.height, self.width = input_size
        self.num_patches = self.depth * self.height * self.width

    def get_shape(self) -> tuple[int, int, int]:
        return self.depth, self.height, self.width

    def __call__(self, num_masking_patches: int = 0) -> np.ndarray:
        mask = np.zeros(self.get_shape(), dtype=bool)
        if num_masking_patches > 0:
            indices = random.sample(range(self.num_patches), k=num_masking_patches)
            mask.reshape(-1)[indices] = True
        return mask
