from __future__ import annotations

import numpy as np

import torch

from koine_machines.inference.infer import Block, OmeZarrBlockDataset, normalize_robust_batch


class _StubReader:
    def __init__(self, patch: np.ndarray) -> None:
        self.patch = patch

    def read(self, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        assert (y0, x0, h, w) == (0, 0, 4, 4)
        return self.patch.copy()


def test_block_dataset_flags_all_zero_patches_for_skipping() -> None:
    block = Block(y0=0, x0=0, valid_h=4, valid_w=4)
    empty = np.zeros((4, 4, 3), dtype=np.uint8)
    occupied = empty.copy()
    occupied[1, 2, 1] = 7

    empty_dataset = OmeZarrBlockDataset(
        reader=_StubReader(empty),
        blocks=[block],
        patch_size=4,
    )
    occupied_dataset = OmeZarrBlockDataset(
        reader=_StubReader(occupied),
        blocks=[block],
        patch_size=4,
    )

    _, meta_empty = empty_dataset[0]
    _, meta_occupied = occupied_dataset[0]

    assert meta_empty.tolist() == [0, 0, 4, 4, 0]
    assert meta_occupied.tolist() == [0, 0, 4, 4, 1]


def test_normalize_robust_batch_matches_numpy_normalize_robust() -> None:
    from vesuvius.image_proc.intensity.normalization import normalize_robust

    rng = np.random.default_rng(0)
    patches = [
        rng.integers(0, 256, (17, 64, 64), dtype=np.uint8),
        np.full((17, 64, 64), 7, dtype=np.uint8),  # flat patch: exercises the fallback scale
    ]
    for patch in patches:
        expected = normalize_robust(patch)
        actual = normalize_robust_batch(torch.from_numpy(patch)[None])[0].numpy()
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


def test_block_dataset_defers_robust_normalization_when_asked() -> None:
    block = Block(y0=0, x0=0, valid_h=4, valid_w=4)
    patch = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    dataset = OmeZarrBlockDataset(
        reader=_StubReader(patch),
        blocks=[block],
        patch_size=4,
        preprocessing="tifxyz_robust",
        defer_robust_normalization=True,
    )
    image, meta = dataset[0]
    assert image.dtype == torch.uint8
    assert image.shape == (1, 3, 4, 4)
    assert int(meta[4]) == 1
