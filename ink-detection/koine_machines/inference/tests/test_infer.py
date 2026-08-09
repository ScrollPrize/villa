from __future__ import annotations

import numpy as np

from koine_machines.inference.infer import Block, OmeZarrBlockDataset


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
