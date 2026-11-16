"""Flat inference leaves pixels without surface data empty."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import vesuvius.ink_detection.inference.infer as infer


class ConstantModel(nn.Module):
    """Predict the same probability everywhere, as a model does on blank input."""

    def forward(self, image):
        return torch.full((image.shape[0], 1, 128, 128), 2.0)


def _run(ct, *, emit_surface_pixels, loader_factory):
    reader = infer.FlatPatchReader(
        input_path="fixture", resolution="0", depth_axis_first=True, height=80, width=80,
        layer_indices=np.arange(17), output_depth=17, preprocessing="tifxyz_robust",
    )
    reader._array = ct
    blocks = infer.iter_blocks((80, 80), 128, 64)
    dataset = infer.FlatBlockDataset(
        reader=reader, blocks=blocks, patch_size=128, preprocessing="tifxyz_robust",
        emit_surface_pixels=emit_surface_pixels,
    )
    out = np.zeros((80, 80), np.float32)
    weight = np.zeros_like(out)
    accumulator = infer.ChunkAccumulator(
        shape=out.shape, chunk_shape=(32, 32), prob_sum_store=out, weight_sum_store=weight,
        contribution_counts=infer.compute_chunk_contribution_counts(blocks, chunk_shape=(32, 32)),
    )
    infer.run_block_inference(
        loader=loader_factory(dataset), model=ConstantModel(), accumulator=accumulator,
        weight_map=np.ones((128, 128), np.float32), mask=None, device=torch.device("cpu"),
        amp_dtype=None, tta_axes=(), tta_batch_size=None,
    )
    accumulator.flush_remaining()
    return next(infer.iter_probability_tiles(out, weight, (80, 80))), weight


def _real_ct_with_hole():
    with np.load(Path(__file__).parent / "fixtures/ink_roi_case.npz") as saved:
        ct = np.array(saved["ct"])
    ct[ct == 0] = 1  # every pixel starts on the surface
    off_surface = np.zeros((80, 80), bool)
    off_surface[:30, 50:] = True  # a corner the mesh does not cover
    ct[:, off_surface] = 0
    return ct, off_surface


def test_pixels_without_surface_stay_empty():
    ct, off_surface = _real_ct_with_hole()
    for loader_factory in (
        lambda dataset: DataLoader(dataset, batch_size=1),
        lambda dataset: infer.ThreadedBatchLoader(dataset, batch_size=2, num_threads=2, prefetch_batches=1),
    ):
        image, weight = _run(ct, emit_surface_pixels=True, loader_factory=loader_factory)
        assert not image[off_surface].any() and not weight[off_surface].any()
        assert (image[~off_surface] > 200).all()


def test_predict_off_surface_keeps_the_previous_output():
    ct, off_surface = _real_ct_with_hole()
    image, _ = _run(ct, emit_surface_pixels=False, loader_factory=lambda dataset: DataLoader(dataset, batch_size=1))
    assert (image > 200).all()


def test_cli_flag():
    assert not infer.parse_args(["input.zarr", "model.pth", "out.tif"]).predict_off_surface
    assert infer.parse_args(["input.zarr", "model.pth", "out.tif", "--predict-off-surface"]).predict_off_surface
