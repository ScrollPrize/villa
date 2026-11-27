"""--mask-off-surface leaves pixels without surface data empty."""
from pathlib import Path

import numpy as np
import pytest
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
        input_path="fixture", resolution="0", depth_axis_first=True, height=ct.shape[1], width=ct.shape[2],
        layer_indices=np.arange(17), output_depth=17, preprocessing="tifxyz_robust",
    )
    reader._array = ct
    blocks = infer.iter_blocks(ct.shape[1:], 128, 64)
    dataset = infer.FlatBlockDataset(
        reader=reader, blocks=blocks, patch_size=128, preprocessing="tifxyz_robust",
        emit_surface_pixels=emit_surface_pixels,
    )
    out = np.zeros(ct.shape[1:], np.float32)
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
    return next(infer.iter_probability_tiles(out, weight, out.shape)), weight, len(blocks)


def _real_ct_with_hole():
    with np.load(Path(__file__).parent / "fixtures/ink_roi_case.npz") as saved:
        ct = np.array(saved["ct"])
    ct[ct == 0] = 1  # every pixel starts on the surface
    ct = np.tile(ct, (1, 4, 4))  # 320 x 320: several tiles per batch
    off_surface = np.zeros(ct.shape[1:], bool)
    off_surface[:128, :128] = True  # one whole tile off the mesh, which is skipped
    off_surface[150:190, 200:320] = True  # a notch at the edge
    off_surface[250:260, 40:50] = True  # a hole inside
    ct[:, off_surface] = 0
    return ct, off_surface


def test_pixels_without_surface_stay_empty():
    ct, off_surface = _real_ct_with_hole()
    for loader_factory in (
        lambda dataset: DataLoader(dataset, batch_size=4),
        lambda dataset: infer.ThreadedBatchLoader(dataset, batch_size=4, num_threads=2, prefetch_batches=1),
    ):
        image, weight, block_count = _run(ct, emit_surface_pixels=True, loader_factory=loader_factory)
        assert block_count > 4  # batches mix the skipped tile with forwarded ones
        assert not image[off_surface].any() and not weight[off_surface].any()
        assert (image[~off_surface] > 200).all()


def test_default_writes_scheduled_tiles_in_full():
    ct, off_surface = _real_ct_with_hole()
    image, _, _ = _run(ct, emit_surface_pixels=False, loader_factory=lambda dataset: DataLoader(dataset, batch_size=4))
    assert (image[~off_surface] > 200).all()
    assert (image[150:190, 200:320] > 200).all() and (image[250:260, 40:50] > 200).all()


def test_third_loader_column_must_be_a_surface_map():
    ct, _ = _real_ct_with_hole()

    def loader_factory(dataset):
        return [(image[None], metadata[None], metadata[None]) for image, metadata, _ in dataset]

    with pytest.raises(TypeError, match="surface"):
        _run(ct, emit_surface_pixels=True, loader_factory=loader_factory)


def test_cli_flag():
    assert not infer.parse_args(["input.zarr", "model.pth", "out.tif"]).mask_off_surface
    assert infer.parse_args(["input.zarr", "model.pth", "out.tif", "--mask-off-surface"]).mask_off_surface
