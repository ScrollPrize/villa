"""SlidingWindowDataset keeps its tile boxes as a tensor, which spawned DataLoader workers share
instead of receiving a pickled copy, and still hands out numpy int32 rows like before."""

import sys
from multiprocessing.reduction import ForkingPickler
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import inference  # noqa: E402

TILE = 16


def make_dataset(boxes):
    layers = np.zeros((TILE, TILE, 2), dtype=np.uint8)  # the boxes are never read in these tests
    return inference.SlidingWindowDataset(inference.LayersSource(layers), boxes, False, TILE, 200)


def make_boxes(n):
    x = np.arange(n, dtype=np.int64) * 4
    return np.stack([x, x + 1, x + TILE, x + 1 + TILE], axis=1)  # int64, as np.asarray(list of lists) gives


def test_boxes_are_an_int32_tensor_with_the_input_values():
    boxes = make_boxes(50)
    ds = make_dataset(boxes)
    assert isinstance(ds.xyxys, torch.Tensor) and ds.xyxys.dtype == torch.int32
    assert len(ds) == 50 and np.array_equal(ds.xyxys.numpy(), boxes)


def test_item_coordinates_are_int32_numpy_rows_with_the_input_values():
    boxes = make_boxes(50)
    ds = make_dataset(boxes)
    for i in (0, 7, 49):
        xy = ds[i][1]
        assert isinstance(xy, np.ndarray) and xy.dtype == np.int32 and np.array_equal(xy, boxes[i])


def test_non_contiguous_boxes_are_accepted():
    boxes = make_boxes(50)
    wide = np.zeros((50, 8), dtype=np.int64)
    wide[:, ::2] = boxes  # every other column: not contiguous
    ds = make_dataset(wide[:, ::2])
    assert np.array_equal(ds.xyxys.numpy(), boxes)


def test_start_up_message_for_a_worker_does_not_grow_with_the_tile_count():
    # What multiprocessing's spawn start method sends each worker: the dataset pickled with ForkingPickler.
    small = len(ForkingPickler.dumps(make_dataset(make_boxes(100))))
    large = len(ForkingPickler.dumps(make_dataset(make_boxes(20000))))  # 320 KB of boxes as int32
    assert large < 16 * 1024 and large - small < 1024
