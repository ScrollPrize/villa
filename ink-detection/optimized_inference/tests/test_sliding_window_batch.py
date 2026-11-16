"""Reading a batch's tiles as row runs (SlidingWindowDataset.__getitems__) must return exactly the
tiles that reading them one by one (__getitem__) returns, also through a DataLoader, and a batch
that wraps a row boundary or holds far-apart clusters must be read as several small regions."""

import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, default_collate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import inference  # noqa: E402

TILE, STRIDE = 32, 8
HEIGHT, WIDTH, DEPTH = 100, 90, 6
COLS = len(inference._grid_1d(WIDTH, TILE, STRIDE))  # tiles per row of the grid below


class SpySource(inference.LayersSource):
    """A LayersSource that records every read_roi call as (y1, y2, x1, x2)."""

    def __init__(self, layers):
        super().__init__(layers)
        self.calls = []

    def read_roi(self, y1, y2, x1, x2):
        self.calls.append((y1, y2, x1, x2))
        return super().read_roi(y1, y2, x1, x2)


def make_layers(height=HEIGHT, width=WIDTH, depth=DEPTH):
    layers = np.random.default_rng(0).integers(0, 256, size=(height, width, depth), dtype=np.uint8)
    layers[:30, :20] = 0  # no surface there: those pixels must come out invalid
    return layers


def make_boxes():
    return np.asarray([[x, y, x + TILE, y + TILE]
                       for y in inference._grid_1d(HEIGHT, TILE, STRIDE)
                       for x in inference._grid_1d(WIDTH, TILE, STRIDE)])


def make_dataset(reverse=False, source=None, boxes=None, size=TILE):
    source = source if source is not None else inference.LayersSource(make_layers())
    boxes = boxes if boxes is not None else make_boxes()
    return inference.SlidingWindowDataset(source, boxes, reverse, size, 200)


def assert_samples_equal(got_samples, want_samples):
    """Images, boxes and validity masks: same values, dtypes and types."""
    assert len(got_samples) == len(want_samples)
    for got_sample, want_sample in zip(got_samples, want_samples):
        assert len(got_sample) == len(want_sample) == 3
        for got, want in zip(got_sample, want_sample):
            assert type(got) is type(want)
            got, want = np.asarray(got), np.asarray(want)
            assert got.dtype == want.dtype and got.shape == want.shape and np.array_equal(got, want)


@pytest.mark.parametrize("reverse", [False, True])
def test_batch_region_matches_per_tile(reverse):
    ds = make_dataset(reverse)
    indices = list(range(3, 40))  # spans several tile rows
    assert_samples_equal(ds.__getitems__(indices), [ds[i] for i in indices])


@pytest.mark.parametrize("reverse", [False, True])
def test_dataloader_batches_match_per_tile(reverse):
    ds = make_dataset(reverse)
    batch_size = 7
    assert len(ds) % batch_size != 0  # the last batch is shorter
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    n_batches = 0
    for b, (images, xys, valids) in enumerate(loader):
        batch_indices = list(range(b * batch_size, min((b + 1) * batch_size, len(ds))))
        want_images, want_xys, want_valids = default_collate([ds[i] for i in batch_indices])
        assert images.dtype == want_images.dtype and torch.equal(images, want_images)
        assert xys.dtype == want_xys.dtype == torch.int32 and torch.equal(xys, want_xys)
        assert valids.dtype == want_valids.dtype and torch.equal(valids, want_valids)
        n_batches += 1
    assert n_batches == -(-len(ds) // batch_size)


@pytest.mark.skipif("spawn" not in multiprocessing.get_all_start_methods(), reason="no spawn start method")
def test_two_spawned_workers_match_zero_workers():
    ds = make_dataset(reverse=True)
    kwargs = dict(batch_size=7, shuffle=False)
    want = list(DataLoader(ds, num_workers=0, **kwargs))
    got = list(DataLoader(ds, num_workers=2, multiprocessing_context="spawn", timeout=300, **kwargs))
    assert len(got) == len(want) == -(-len(ds) // 7)
    for got_batch, want_batch in zip(got, want):
        for g, w in zip(got_batch, want_batch):
            assert g.dtype == w.dtype and torch.equal(g, w)


def test_one_row_is_one_read():
    spy = SpySource(make_layers())
    ds = make_dataset(False, spy)
    ds.__getitems__(list(range(3, COLS)))  # the tail of the first row
    assert len(spy.calls) == 1
    y1, y2, x1, x2 = spy.calls[0]
    assert (y2 - y1, x1, x2) == (TILE, make_boxes()[3][0], WIDTH)


def test_dense_batch_reads_one_region_per_row():
    spy = SpySource(make_layers())
    ds = make_dataset(False, spy)
    indices = list(range(5, 3 * COLS + 4))  # rows 0..3, the first and the last one partial
    got = ds.__getitems__(indices)
    assert len(spy.calls) == 4
    assert all(y2 - y1 == TILE for y1, y2, _, _ in spy.calls)
    assert_samples_equal(got, [ds[i] for i in indices])


@pytest.mark.parametrize("tail", [1, 4])
def test_batch_that_wraps_a_row_boundary_is_read_as_two_regions(tail):
    spy = SpySource(make_layers())
    ds = make_dataset(False, spy)
    first = 2 * COLS + COLS - tail  # the last `tail` tiles of row 2, then the first 3 of row 3
    indices = list(range(first, 3 * COLS + 3))
    got = ds.__getitems__(indices)
    assert len(spy.calls) == 2
    assert all(y2 - y1 == TILE for y1, y2, _, _ in spy.calls)
    assert_samples_equal(got, [ds[i] for i in indices])


def test_gap_wider_than_a_tile_starts_a_new_run():
    spy = SpySource(make_layers())
    ds = make_dataset(False, spy)
    boxes = make_boxes()
    indices = [0, 1, COLS - 1]  # two tiles at the left edge and one at the right edge of row 0
    assert boxes[COLS - 1][0] - boxes[1][0] > TILE
    got = ds.__getitems__(indices)
    assert len(spy.calls) == 2
    assert all(x2 - x1 < WIDTH for _, _, x1, x2 in spy.calls)  # no read spans the gap
    assert_samples_equal(got, [ds[i] for i in indices])


def test_tile_on_another_row_starts_a_new_run_even_if_it_lies_to_the_right():
    spy = SpySource(make_layers())
    ds = make_dataset(False, spy)
    indices = [0, COLS + 1]  # x1 grows by 8, y1 changes by 8
    got = ds.__getitems__(indices)
    assert len(spy.calls) == 2 and all(y2 - y1 == TILE for y1, y2, _, _ in spy.calls)
    assert_samples_equal(got, [ds[i] for i in indices])


def test_tiles_in_opposite_corners_are_read_as_two_small_regions():
    spy = SpySource(make_layers())
    ds = make_dataset(False, spy)
    indices = [0, 1, len(ds) - 2, len(ds) - 1]
    got = ds.__getitems__(indices)
    assert len(spy.calls) == 2
    for y1, y2, x1, x2 in spy.calls:  # neither read comes anywhere near the whole image
        assert y2 - y1 == TILE and x2 - x1 <= TILE + STRIDE
    assert_samples_equal(got, [ds[i] for i in indices])


def test_two_far_apart_clusters_are_never_read_as_one_rectangle():
    tile, stride, width, rows, cols = 64, 16, 6000, 8, 16
    layers = make_layers(height=tile + (rows - 1) * stride, width=width, depth=3)
    left = [x for x in range(0, cols * stride, stride)]  # x1 0..240
    right = [x for x in range(5600, 5600 + cols * stride, stride)]  # x1 5600..5840
    boxes = np.asarray([[x, y, x + tile, y + tile] for y in range(0, rows * stride, stride) for x in left + right])
    spy = SpySource(layers)
    ds = make_dataset(False, spy, boxes=boxes, size=tile)
    indices = list(range(len(ds)))  # 256 tiles in sampler order
    got = ds.__getitems__(indices)
    assert len(spy.calls) >= 2
    for _, _, x1, x2 in spy.calls:
        assert x2 <= left[-1] + tile or x1 >= right[0]  # inside one cluster
    assert_samples_equal(got, [ds[i] for i in indices])


def test_run_whose_region_exceeds_its_tiles_is_read_tile_by_tile():
    spy = SpySource(make_layers())
    ds = make_dataset(False, spy)
    boxes = make_boxes()
    indices = [0, COLS - 1]  # the run rule never builds this run; the safety check still has to hold
    got = ds._read_run(indices, boxes[indices])
    assert len(spy.calls) == 2 and all((y2 - y1, x2 - x1) == (TILE, TILE) for y1, y2, x1, x2 in spy.calls)
    assert_samples_equal(got, [ds[i] for i in indices])


def test_tiles_that_need_resizing_are_read_per_tile():
    spy = SpySource(make_layers())
    ds = make_dataset(False, spy, size=TILE // 2)  # network input smaller than the tile: cv2.resize
    indices = list(range(3, 20))
    got = ds.__getitems__(indices)
    assert len(spy.calls) == len(indices)
    assert all((y2 - y1, x2 - x1) == (TILE, TILE) for y1, y2, x1, x2 in spy.calls)
    assert_samples_equal(got, [ds[i] for i in indices])
    assert got[0][0].shape[-2:] == (TILE // 2, TILE // 2)


@pytest.mark.parametrize("reverse", [False, True])
def test_tiles_past_the_source_edge_are_zero_padded_like_per_tile(reverse):
    boxes = np.asarray([[x, y, x + TILE, y + TILE]
                        for y in (-12, HEIGHT - 20)  # above the top edge, below the bottom edge
                        for x in range(-20, WIDTH - 11, STRIDE)])  # left of the left edge ... right of the right edge
    spy = SpySource(make_layers())
    ds = make_dataset(reverse, spy, boxes=boxes)
    indices = list(range(len(ds)))
    got = ds.__getitems__(indices)
    assert len(spy.calls) == 2
    want = [ds[i] for i in indices]
    assert_samples_equal(got, want)
    assert any((sample[2] == 0).any() for sample in got)  # padded pixels are invalid, as per tile
    assert any((sample[0].numpy() == 0).any() for sample in got)


def test_duplicate_and_reordered_indices_return_the_same_samples_in_the_same_order():
    ds = make_dataset(False)
    for indices in ([5, 3, 5, 6, 6, 0, 79, 40, 41, 40], list(range(29, 2, -1)), [20, 3, 11, 3, 3, 28, 1]):
        assert_samples_equal(ds.__getitems__(indices), [ds[i] for i in indices])


def test_returned_boxes_are_int32_numpy_rows():
    ds = make_dataset(False)
    boxes = make_boxes()
    dense = [5, 3, 5, 6]  # unordered, with a duplicate
    sparse = [0, len(ds) - 1]  # opposite corners
    for i in dense + sparse:
        xy = ds[i][1]
        assert isinstance(xy, np.ndarray) and xy.dtype == np.int32 and np.array_equal(xy, boxes[i])
    for indices in (dense, sparse):
        for i, sample in zip(indices, ds.__getitems__(indices)):
            xy = sample[1]
            assert isinstance(xy, np.ndarray) and xy.dtype == np.int32 and np.array_equal(xy, boxes[i])
