from __future__ import annotations

import unittest

import numpy as np

from ink.recipes.data.patching import build_patch_index


def _label_tile_is_empty(label_tile) -> bool:
    tile = np.asarray(label_tile)
    if tile.size == 0:
        return True
    if np.issubdtype(tile.dtype, np.floating):
        return bool(np.all(tile < 0.01))
    if np.issubdtype(tile.dtype, np.integer):
        return bool(np.all(tile < 3))
    return bool(np.all(tile.astype(np.float32, copy=False) < 0.01))


def _reference_extract_patch_coordinates(
    mask,
    fragment_mask,
    *,
    size: int,
    tile_size: int,
    stride: int,
    filter_empty_tile: bool = False,
) -> tuple[tuple[int, int, int, int], ...]:
    label_mask = np.asarray(mask)
    fragment_mask = np.asarray(fragment_mask)
    max_x = int(fragment_mask.shape[1] - tile_size)
    max_y = int(fragment_mask.shape[0] - tile_size)
    if max_x < 0 or max_y < 0:
        return tuple()

    x1_list = range(0, max_x + 1, stride)
    y1_list = range(0, max_y + 1, stride)
    seen = set()
    xyxys: list[tuple[int, int, int, int]] = []
    for y_tile in y1_list:
        for x_tile in x1_list:
            if filter_empty_tile and _label_tile_is_empty(
                label_mask[y_tile:y_tile + tile_size, x_tile:x_tile + tile_size]
            ):
                continue
            tile = fragment_mask[y_tile:y_tile + tile_size, x_tile:x_tile + tile_size]
            tile_has_invalid = bool(np.any(tile == 0))
            if tile_has_invalid and filter_empty_tile:
                continue
            for yi in range(0, tile_size, size):
                for xi in range(0, tile_size, size):
                    y1 = int(y_tile + yi)
                    x1 = int(x_tile + xi)
                    y2 = int(y1 + size)
                    x2 = int(x1 + size)
                    key = (x1, y1, x2, y2)
                    if key in seen:
                        continue
                    if tile_has_invalid and (not filter_empty_tile) and np.any(fragment_mask[y1:y2, x1:x2] == 0):
                        continue
                    seen.add(key)
                    xyxys.append(key)
    return tuple(xyxys)


class PatchCoordinateExtractionTests(unittest.TestCase):
    def _actual_patch_coordinates(
        self,
        label_mask,
        fragment_mask,
        *,
        size: int,
        tile_size: int,
        stride: int,
        filter_empty_tile: bool = False,
    ) -> tuple[tuple[int, int, int, int], ...]:
        _bbox_rows, xyxys, _sample_bbox_indices = build_patch_index(
            label_mask,
            fragment_mask,
            size=size,
            tile_size=tile_size,
            stride=stride,
            filter_empty_tile=filter_empty_tile,
        )
        return tuple(tuple(int(value) for value in row) for row in np.asarray(xyxys, dtype=np.int64).tolist())

    def test_extract_patch_coordinates_matches_reference_on_sparse_masks(self):
        label_mask = np.full((16, 16), 255, dtype=np.uint8)
        fragment_mask = np.zeros((16, 16), dtype=np.uint8)
        fragment_mask[0:8, 0:8] = 255
        fragment_mask[8:16, 8:16] = 255
        fragment_mask[10:12, 10:12] = 0

        for stride in (2, 4):
            expected = _reference_extract_patch_coordinates(
                label_mask,
                fragment_mask,
                size=4,
                tile_size=8,
                stride=stride,
                filter_empty_tile=False,
            )
            actual = self._actual_patch_coordinates(
                label_mask,
                fragment_mask,
                size=4,
                tile_size=8,
                stride=stride,
                filter_empty_tile=False,
            )
            self.assertEqual(set(actual), set(expected))

    def test_extract_patch_coordinates_matches_reference_on_irregular_components(self):
        label_mask = np.full((20, 20), 255, dtype=np.uint8)
        fragment_mask = np.zeros((20, 20), dtype=np.uint8)
        fragment_mask[2:16, 2:6] = 255
        fragment_mask[12:16, 2:16] = 255
        fragment_mask[4:12, 10:18] = 255
        fragment_mask[6:8, 12:14] = 0

        expected = _reference_extract_patch_coordinates(
            label_mask,
            fragment_mask,
            size=4,
            tile_size=8,
            stride=4,
            filter_empty_tile=False,
        )
        actual = self._actual_patch_coordinates(
            label_mask,
            fragment_mask,
            size=4,
            tile_size=8,
            stride=4,
            filter_empty_tile=False,
        )

        self.assertEqual(set(actual), set(expected))

    def test_extract_patch_coordinates_preserves_filter_empty_tile_behavior(self):
        label_mask = np.zeros((12, 12), dtype=np.uint8)
        label_mask[4:8, 4:8] = 255
        fragment_mask = np.full((12, 12), 255, dtype=np.uint8)

        expected = _reference_extract_patch_coordinates(
            label_mask,
            fragment_mask,
            size=4,
            tile_size=8,
            stride=4,
            filter_empty_tile=True,
        )
        actual = self._actual_patch_coordinates(
            label_mask,
            fragment_mask,
            size=4,
            tile_size=8,
            stride=4,
            filter_empty_tile=True,
        )

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
