"""Regression tests for vectorized subtiling patch discovery and single-read surface planes."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.data.patch_finding_subtiling import (
    _label_tile_is_empty,
    build_patch_index,
    find_segment_patches,
)
from vesuvius.ink_detection.types import Patch, Segment


def _ref_build_patch_index(
    mask: np.ndarray,
    fragment_mask: np.ndarray,
    *,
    size: int,
    tile_size: int,
    stride: int,
) -> np.ndarray:
    label_mask = np.asarray(mask)
    valid_mask = np.asarray(fragment_mask).astype(bool, copy=False)
    seen: set[tuple[int, int, int, int]] = set()
    coords: list[tuple[int, int, int, int]] = []
    max_y = valid_mask.shape[0] - tile_size
    max_x = valid_mask.shape[1] - tile_size
    if max_y >= 0 and max_x >= 0:
        for y_tile in range(0, max_y + 1, stride):
            for x_tile in range(0, max_x + 1, stride):
                if _label_tile_is_empty(label_mask[y_tile : y_tile + tile_size, x_tile : x_tile + tile_size]):
                    continue
                if not bool(valid_mask[y_tile : y_tile + tile_size, x_tile : x_tile + tile_size].all()):
                    continue
                for yi in range(0, tile_size, size):
                    for xi in range(0, tile_size, size):
                        key = (x_tile + xi, y_tile + yi, x_tile + xi + size, y_tile + yi + size)
                        if key not in seen:
                            seen.add(key)
                            coords.append(key)
    return np.asarray(coords, dtype=np.int64).reshape(-1, 4)


def _make_segment(tmp_path: Path, **overrides) -> Segment:
    authored = {
        "mode": "flat",
        "patch_size": [3, 8, 8],
        "patch_overlap": 0.5,
        "patch_min_labeled_coverage": 0.1,
        "patch_finding_type": "subtiling",
        "patch_finding_filter_empty_tile": True,
        "image_normalization": "none",
        "out_dir": str(tmp_path),
        "dataloader_workers": 1,
        "datasets": [{"segments_path": str(tmp_path), "volume_scale": 0}],
    }
    authored.update(overrides)
    cfg = InkDataConfig.from_mapping(authored)
    return Segment(
        data_config=cfg,
        source=cfg.datasets[0],
        dataset_idx=0,
        segment_relpath="seg-a",
        segment_dir=tmp_path / "seg-a",
        segment_name="seg-a",
        image_volume=Path("image"),
        inklabels=Path("ink"),
        supervision_mask=Path("sup"),
    )


@pytest.mark.parametrize(
    "shape,size,tile_size,stride",
    [
        ((64, 64), 8, 8, 4),
        ((64, 64), 8, 16, 4),
        ((65, 53), 4, 12, 4),
        ((65, 53), 4, 12, 6),  # gcd=6 >= 4, overlapping subtiles
        ((57, 49), 4, 12, 3),  # gcd=3 < 4 fallback
        ((57, 49), 4, 8, 5),   # coprime gcd=1 fallback
        ((7, 7), 8, 8, 4),     # smaller than tile_size -> empty
    ],
)
@pytest.mark.parametrize("mask_dtype", [np.uint8, np.int16, np.float32, np.bool_])
def test_build_patch_index_bit_identical_all_dtypes_and_strides(shape, size, tile_size, stride, mask_dtype):
    rng = np.random.default_rng(100 + size + stride)
    if mask_dtype == np.uint8:
        mask = rng.choice([0, 1, 2, 3, 255], size=shape, p=[0.7, 0.1, 0.1, 0.05, 0.05]).astype(np.uint8)
    elif mask_dtype == np.int16:
        mask = rng.choice([-5, 0, 2, 3, 10], size=shape, p=[0.2, 0.5, 0.2, 0.05, 0.05]).astype(np.int16)
    elif mask_dtype == np.float32:
        mask = rng.choice([0.0, 0.005, 0.01, 0.5, -1.0], size=shape, p=[0.7, 0.1, 0.05, 0.05, 0.1]).astype(np.float32)
        if mask.size > 10:
            mask.ravel()[7] = np.nan
            mask.ravel()[9] = -np.inf
    else:
        mask = rng.random(size=shape) < 0.05

    frag = (rng.random(size=shape) > 0.03).astype(np.uint8)
    _, got, idx = build_patch_index(
        mask, frag, size=size, tile_size=tile_size, stride=stride, filter_empty_tile=True
    )
    ref = _ref_build_patch_index(mask, frag, size=size, tile_size=tile_size, stride=stride)
    np.testing.assert_array_equal(got, ref)
    np.testing.assert_array_equal(idx, np.full((len(ref),), -1, dtype=np.int32))


def test_build_patch_index_signed_negative_fragment_mask():
    # Signed integer fragment_mask with negative nonzero entries must be treated as valid (truthy).
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2, 2] = 5
    frag = np.full((16, 16), -1, dtype=np.int8)
    frag[0, 15] = 0
    _, got, _ = build_patch_index(mask, frag, size=4, tile_size=8, stride=4, filter_empty_tile=True)
    ref = _ref_build_patch_index(mask, frag, size=4, tile_size=8, stride=4)
    np.testing.assert_array_equal(got, ref)


def test_subtiling_patch_finding_reads_each_mask_surface_once(tmp_path):
    seg = replace(
        _make_segment(tmp_path, patch_size=[3, 8, 8], patch_finding_tile_size=16, patch_finding_stride=4),
        validation_mask=Path("val"),
    )
    rng = np.random.default_rng(7)
    ink = np.zeros((3, 64, 64), dtype=np.uint8)
    ink[1] = (rng.random((64, 64)) < 0.08).astype(np.uint8) * 10
    sup = np.ones((3, 64, 64), dtype=np.uint8)
    val = np.zeros((3, 64, 64), dtype=np.uint8)
    val[1, 16:40, 16:40] = 1
    img = np.zeros((3, 64, 64), dtype=np.uint8)

    reads: dict[str, int] = {"ink": 0, "sup": 0, "val": 0}

    class Counted(np.ndarray):
        def __new__(cls, arr, tag):
            obj = np.asarray(arr).view(cls)
            obj._tag = tag
            return obj

        def __getitem__(self, key):
            reads[self._tag] += 1
            return np.asarray(self)[key]

    vols = {
        "image": img,
        "ink": Counted(ink, "ink"),
        "sup": Counted(sup, "sup"),
        "val": Counted(val, "val"),
    }
    train, held = find_segment_patches(seg, lambda p, _: vols[str(p)])
    assert len(train) > 0 and len(held) > 0
    assert reads == {"ink": 1, "sup": 1, "val": 1}


@pytest.mark.parametrize("sup_val,val_val", [(1, 1), (255, 1)])
def test_find_segment_patches_matches_bitwise_validation_and_coverage(tmp_path, sup_val, val_val):
    seg = replace(
        _make_segment(
            tmp_path,
            patch_size=[3, 8, 8],
            patch_finding_tile_size=16,
            patch_finding_stride=4,
            patch_min_labeled_coverage=0.2,
        ),
        validation_mask=Path("val"),
    )
    rng = np.random.default_rng(42)
    ink = np.zeros((3, 48, 48), dtype=np.uint8)
    ink[1] = rng.choice([0, 2, 5], size=(48, 48), p=[0.8, 0.1, 0.1]).astype(np.uint8)
    sup = np.full((3, 48, 48), sup_val, dtype=np.uint8)
    val = np.zeros((3, 48, 48), dtype=np.uint8)
    val[1, 8:28, 8:28] = val_val
    img = np.zeros((3, 48, 48), dtype=np.uint8)
    vols = {"image": img, "ink": ink, "sup": sup, "val": val}

    got_train, got_val = find_segment_patches(seg, lambda p, _: vols[str(p)])

    # Reference computation matching original per-patch logic
    xyxys = _ref_build_patch_index(ink[1], sup[1], size=8, tile_size=16, stride=4)
    ref_train: list[Patch] = []
    ref_val: list[Patch] = []
    for x1, y1, _, _ in xyxys.tolist():
        bbox = (0, int(y1), int(x1), 3, int(y1) + 8, int(x1) + 8)
        sp = sup[1, y1 : y1 + 8, x1 : x1 + 8]
        vp = val[1, y1 : y1 + 8, x1 : x1 + 8]
        has_t = bool(sp.size and np.any(sp))
        has_v = bool(vp.size and np.any(vp))
        if has_t and has_v:
            has_t = bool(np.any(sp & ~vp))
        if has_v:
            ref_val.append(Patch(segment=seg, bbox=bbox, is_validation=True, supervision_mask_override=seg.validation_mask))
        lp = ink[1, y1 : y1 + 8, x1 : x1 + 8]
        ly, lx = np.nonzero(lp)
        cov = float((int(ly.max()) - int(ly.min()) + 1) * (int(lx.max()) - int(lx.min()) + 1)) / float(lp.size) if ly.size else 0.0
        if has_t and cov >= 0.2:
            ref_train.append(Patch(segment=seg, bbox=bbox))

    assert got_train == ref_train
    assert got_val == ref_val
