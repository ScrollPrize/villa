"""A PyTorch stand-in for ``inkrs.Engine``, so tests run villa's inkrs adapter without inkrs."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from vesuvius.ink_detection.inference import infer
from vesuvius.ink_detection.inference.engines import InkrsEngine

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


class TorchStandInError(RuntimeError):
    """The stand-in's counterpart of ``inkrs.InkrsError``."""


class TorchStandInUnsupported(TorchStandInError):
    """The stand-in's counterpart of ``inkrs.UnsupportedError``."""


class TorchStandInEngine:
    """PyTorch behind the ``inkrs.Engine`` signatures the villa adapter calls.

    It rebuilds villa's tile schedule from the occupancy mask it is given,
    as inkrs does, then runs villa's own PyTorch forward and blend. Through
    the inkrs adapter it should reproduce PyTorch exactly, which checks the
    adapter's window, layer order and occupancy hand-off on real data
    without inkrs installed.
    """

    def __init__(
        self,
        checkpoint: Any,
        *,
        device: int | str | torch.device = 0,
        batch_size: int = 64,
        amp_dtype: str = "auto",
    ) -> None:
        self.configured = infer.configure_single_model(
            Path(checkpoint), amp_dtype=amp_dtype
        )
        self.device = (
            torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        )
        self.model = self.configured.model.to(self.device).eval()
        self.batch_size = int(batch_size)

    def run(
        self,
        volume: np.ndarray,
        windows: Sequence[tuple[int, int]],
        *,
        stride: int | None = None,
        occupancy: Any = "auto",
        coverage: bool = False,
        outputs: Any = None,
    ) -> list[Any]:
        if outputs is not None:
            raise TorchStandInError("the stand-in returns arrays only")
        volume = np.asarray(volume)
        if volume.ndim != 3:
            raise ValueError(f"need a [Z, H, W] volume, got shape {volume.shape}")
        _, height, width = volume.shape
        patch, depth = self.configured.patch_size, self.configured.input_depth
        # inkrs's default: half the loaded reader's tile edge (64, or 192 for reader-v3)
        if stride is None:
            stride = patch // 2
        # the input refusals inkrs documents as UnsupportedError
        if volume.dtype != np.uint8:
            raise TorchStandInUnsupported(f"uint8 volumes only, got {volume.dtype}")
        if min(height, width) < patch:
            raise TorchStandInUnsupported(f"both sides must be at least {patch} px")
        if isinstance(occupancy, str):
            # "auto" means no scan for an in-memory volume; a path is not supported here
            if occupancy != "auto":
                raise TorchStandInError("the stand-in takes an in-memory occupancy array only")
            occupancy = None
        mask, scale = None, (1, 1)
        if occupancy is not None:
            mask = infer.compute_nonempty_mask_from_lowres_array(np.asarray(occupancy))
            scale = (
                max(1, round(height / max(1, mask.shape[0]))),
                max(1, round(width / max(1, mask.shape[1]))),
            )
        blocks = infer.iter_blocks((height, width), patch, stride, mask, scale)
        results = []
        for first, step in windows:
            window = np.ascontiguousarray(volume[first + step * np.arange(depth)])
            probability, covered = self._predict(window, blocks, stride)
            results.append((probability, covered.astype(np.uint8)) if coverage else probability)
        return results

    def _predict(self, window: np.ndarray, blocks: list, stride: int):
        _, height, width = window.shape
        shape = (height, width)
        patch = self.configured.patch_size
        preprocessing = self.configured.preprocessing
        dataset = infer.FlatBlockDataset(
            reader=_ArrayPatchReader(window),
            blocks=blocks,
            patch_size=patch,
            preprocessing=preprocessing,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        chunks = (min(patch, height), min(patch, width))
        probability_sum = np.zeros(shape, dtype=np.float32)
        weight_sum = np.zeros(shape, dtype=np.float32)
        accumulator = infer.ChunkAccumulator(
            shape=shape,
            chunk_shape=chunks,
            prob_sum_store=probability_sum,
            weight_sum_store=weight_sum,
            contribution_counts=infer.compute_chunk_contribution_counts(
                blocks, chunk_shape=chunks
            ),
        )
        infer.run_block_inference(
            loader=loader,
            model=self.model,
            accumulator=accumulator,
            weight_map=infer.compute_importance_map_2d(
                patch_size=(patch, patch), mode="hann"
            ).numpy(),
            mask=None,
            device=self.device,
            amp_dtype=self.configured.amp_dtype,
            tta_axes=(),
            tta_batch_size=None,
        )
        accumulator.flush_remaining()
        probability = next(
            infer.iter_probability_tiles(probability_sum, weight_sum, shape)
        )
        return probability, weight_sum > COVERAGE_EPS


def load_torch_stand_in(
    args: argparse.Namespace, configured_model: Any, device: torch.device
) -> InkrsEngine:
    """The inkrs adapter over the PyTorch stand-in, at villa's batch size.

    It skips InkrsEngine.load's CUDA check so the plumbing also runs on CPU.
    """

    module = SimpleNamespace(
        __version__="torch-stand-in",
        InkrsError=TorchStandInError,
        UnsupportedError=TorchStandInUnsupported,
        Engine=TorchStandInEngine,
    )
    handle = TorchStandInEngine(
        args.checkpoint, device=device, batch_size=args.batch_size, amp_dtype=args.amp_dtype
    )
    return InkrsEngine(module, handle, device_index=0)
