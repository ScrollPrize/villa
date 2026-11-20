"""A PyTorch stand-in for the ``inkrs`` module, so tests run villa's inkrs path without inkrs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from vesuvius.ink_detection.inference import infer

# villa's accumulator counts a pixel as covered above this weight sum
COVERAGE_EPS = 1e-6


class _ArrayPatchReader:
    """FlatPatchReader's ``read`` over an in-memory (Z, H, W) window."""

    def __init__(self, window: np.ndarray) -> None:
        self.window = window

    def read(self, y0: int, x0: int, out_h: int, out_w: int) -> np.ndarray:
        depth, height, width = self.window.shape
        output = np.zeros((out_h, out_w, depth), dtype=np.uint8)
        y1, x1 = min(height, y0 + out_h), min(width, x0 + out_w)
        output[: y1 - y0, : x1 - x0] = np.moveaxis(self.window[:, y0:y1, x0:x1], 0, -1)
        return output


class InkrsError(RuntimeError):
    """The stand-in's ``inkrs.InkrsError``."""


class UnsupportedError(InkrsError):
    """The stand-in's ``inkrs.UnsupportedError``."""


class Engine:
    """villa's own PyTorch forward and blend behind ``inkrs.Engine``'s signatures.

    It rebuilds villa's tile schedule from the occupancy mask it is given, as
    inkrs does, so through villa's inkrs path it reproduces PyTorch exactly,
    which checks the window, layer order and occupancy villa hands over.
    """

    runs = 0

    def __init__(self, checkpoint: Any, *, device: int = 0, batch_size: int = 2) -> None:
        self.configured = infer.configure_single_model(Path(checkpoint), amp_dtype="auto")
        self.model = self.configured.model.eval()
        self.batch_size = batch_size

    def run(
        self,
        volume: np.ndarray,
        windows: Sequence[tuple[int, int]],
        *,
        stride: int,
        occupancy: np.ndarray | None,
        coverage: bool,
    ) -> list[Any]:
        Engine.runs += 1
        _, height, width = volume.shape
        if volume.dtype != np.uint8:
            raise UnsupportedError(f"uint8 volumes only, got {volume.dtype}")
        patch, depth = self.configured.patch_size, self.configured.input_depth
        mask, scale = None, (1, 1)
        if occupancy is not None:
            mask = infer.compute_nonempty_mask_from_lowres_array(np.asarray(occupancy))
            scale = (max(1, round(height / mask.shape[0])), max(1, round(width / mask.shape[1])))
        blocks = infer.iter_blocks((height, width), patch, stride, mask, scale)
        results = []
        for first, step in windows:
            window = np.ascontiguousarray(volume[first + step * np.arange(depth)])
            probability, covered = self._predict(window, blocks)
            results.append((probability, covered.astype(np.uint8)) if coverage else probability)
        return results

    def _predict(self, window: np.ndarray, blocks: list) -> tuple[np.ndarray, np.ndarray]:
        _, height, width = window.shape
        shape = (height, width)
        patch = self.configured.patch_size
        dataset = infer.FlatBlockDataset(
            reader=_ArrayPatchReader(window),
            blocks=blocks,
            patch_size=patch,
            preprocessing=self.configured.preprocessing,
        )
        chunks = (min(patch, height), min(patch, width))
        probability_sum = np.zeros(shape, dtype=np.float32)
        weight_sum = np.zeros(shape, dtype=np.float32)
        accumulator = infer.ChunkAccumulator(
            shape=shape,
            chunk_shape=chunks,
            prob_sum_store=probability_sum,
            weight_sum_store=weight_sum,
            contribution_counts=infer.compute_chunk_contribution_counts(blocks, chunk_shape=chunks),
        )
        infer.run_block_inference(
            loader=DataLoader(dataset, batch_size=self.batch_size, shuffle=False),
            model=self.model,
            accumulator=accumulator,
            weight_map=infer.compute_importance_map_2d(patch_size=(patch, patch), mode="hann").numpy(),
            mask=None,
            device=torch.device("cpu"),
            amp_dtype=self.configured.amp_dtype,
            tta_axes=(),
            tta_batch_size=None,
        )
        accumulator.flush_remaining()
        probability = next(infer.iter_probability_tiles(probability_sum, weight_sum, shape))
        return probability, weight_sum > COVERAGE_EPS


def stand_in_module(engine: Any = Engine) -> SimpleNamespace:
    """An ``inkrs`` module whose ``Engine`` is ``engine``."""

    return SimpleNamespace(
        __version__="stand-in", InkrsError=InkrsError, UnsupportedError=UnsupportedError, Engine=engine
    )
