from __future__ import annotations

import math
import random

import numpy as np


class MaskingGenerator3d:
    def __init__(
        self,
        input_size: int | tuple[int, int, int],
        *,
        mode: str = "block",
        min_num_patches: int = 4,
        max_num_patches: int | None = None,
        min_aspect: float = 0.3,
        max_aspect: float | None = None,
        max_attempts: int = 32,
    ) -> None:
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size, input_size)
        self.depth, self.height, self.width = input_size
        self.num_patches = self.depth * self.height * self.width
        self.mode = mode
        self.min_num_patches = max(1, int(min_num_patches))
        self.max_num_patches = (
            self.num_patches if max_num_patches is None else min(int(max_num_patches), self.num_patches)
        )
        self.max_attempts = max(1, int(max_attempts))

        if min_aspect <= 0:
            raise ValueError("min_aspect must be positive")
        if max_aspect is None:
            max_aspect = 1 / min_aspect
        if max_aspect <= 0:
            raise ValueError("max_aspect must be positive")
        if self.mode not in {"block", "random"}:
            raise ValueError(f"Unsupported masking mode: {self.mode}")
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def get_shape(self) -> tuple[int, int, int]:
        return self.depth, self.height, self.width

    def __call__(self, num_masking_patches: int = 0) -> np.ndarray:
        num_masking_patches = max(0, min(int(num_masking_patches), self.num_patches))
        mask = np.zeros(self.get_shape(), dtype=bool)
        if num_masking_patches <= 0:
            return mask

        if self.mode == "random":
            self._fill_mask_random(mask, num_masking_patches)
            return mask

        mask_count = 0
        while mask_count < num_masking_patches:
            remaining = num_masking_patches - mask_count
            delta = self._mask_block(mask, remaining)
            if delta == 0:
                self._fill_mask_random(mask, remaining)
                break
            mask_count += delta
        return mask

    def _fill_mask_random(self, mask: np.ndarray, num_masking_patches: int) -> None:
        if num_masking_patches <= 0:
            return
        available = np.flatnonzero(~mask.reshape(-1))
        if available.size == 0:
            return
        chosen = random.sample(available.tolist(), k=min(num_masking_patches, available.size))
        mask.reshape(-1)[chosen] = True

    def _mask_block(self, mask: np.ndarray, max_mask_patches: int) -> int:
        max_mask_patches = min(max_mask_patches, self.max_num_patches)
        if max_mask_patches <= 0:
            return 0

        min_mask_patches = min(self.min_num_patches, max_mask_patches)
        for _ in range(self.max_attempts):
            target_volume = random.uniform(min_mask_patches, max_mask_patches)
            depth, height, width = self._sample_block_shape(target_volume)
            if depth > self.depth or height > self.height or width > self.width:
                continue

            start_d = random.randint(0, self.depth - depth)
            start_h = random.randint(0, self.height - height)
            start_w = random.randint(0, self.width - width)
            region = mask[start_d : start_d + depth, start_h : start_h + height, start_w : start_w + width]

            newly_masked = (depth * height * width) - int(region.sum())
            if 0 < newly_masked <= max_mask_patches:
                region[:] = True
                return newly_masked
        return 0

    def _sample_block_shape(self, target_volume: float) -> tuple[int, int, int]:
        depth_scale = math.exp(random.uniform(*self.log_aspect_ratio))
        planar_aspect = math.exp(random.uniform(*self.log_aspect_ratio))

        base = (target_volume / depth_scale) ** (1.0 / 3.0)
        depth = max(1, int(round(base * depth_scale)))
        planar_area = max(target_volume / depth, 1.0)
        height = max(1, int(round(math.sqrt(planar_area * planar_aspect))))
        width = max(1, int(round(math.sqrt(planar_area / planar_aspect))))
        return depth, height, width
