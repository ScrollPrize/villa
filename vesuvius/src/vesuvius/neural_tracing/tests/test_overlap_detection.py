from pathlib import Path

import numpy as np

import vesuvius.neural_tracing.datasets.dataset_rowcol_cond as rowcol_dataset_module
from pherc0139_overlap_test_utils import (
    PARENT_WRAP_NAMES,
    PHERC0139_ROOT,
    cutout_name_map,
    detect_parent_overlap_masks,
    load_parent_tifxyz_segments,
    probe_mask,
    projected_bbox_2d,
)
from vesuvius.neural_tracing.datasets.common import ChunkPatch
from vesuvius.neural_tracing.datasets.dataset_defaults import (
    setdefault_rowcol_cond_dataset_config,
    validate_rowcol_cond_dataset_config,
)
from vesuvius.neural_tracing.datasets.dataset_rowcol_cond import EdtSegDataset


def test_detect_wrap_overlap_masks_marks_overlap_cutouts_and_skips_no_overlap_regions():
    masks = detect_parent_overlap_masks()
    cutouts = cutout_name_map()

    expected_overlap_pairs = {
        cutouts["w018_w019_overlap"]: ("w018_20260130192417321", "w019_20260203030108447"),
        cutouts["w019_w020_overlap"]: ("w020_20260219135139420", "w019_20260203030108447"),
    }
    for cutout_name, parent_names in expected_overlap_pairs.items():
        for parent_name in parent_names:
            stats = probe_mask(masks[parent_name], cutout_name, parent_name)
            assert stats.hit_fraction >= 0.75, (
                f"Expected strong overlap coverage for cutout={cutout_name} in parent={parent_name}, "
                f"required hit_fraction>=0.75, got hit_count={stats.hit_count}/{stats.point_count}, "
                f"hit_fraction={stats.hit_fraction:.6f}, "
                f"distance_median={stats.median_distance:.4f}, distance_p95={stats.p95_distance:.4f}, "
                f"distance_max={stats.max_distance:.4f}"
            )

    no_overlap_cutouts = (cutouts["w018_no_overlap"], cutouts["w019_no_overlap"])
    for cutout_name in no_overlap_cutouts:
        for parent_name in PARENT_WRAP_NAMES:
            stats = probe_mask(masks[parent_name], cutout_name, parent_name)
            assert stats.hit_count == 0, (
                f"Expected no overlap hits for cutout={cutout_name} in parent={parent_name}, "
                f"got hit_count={stats.hit_count}/{stats.point_count}, hit_fraction={stats.hit_fraction:.6f}, "
                f"distance_median={stats.median_distance:.4f}, distance_p95={stats.p95_distance:.4f}, "
                f"distance_max={stats.max_distance:.4f}"
            )


def _build_filter_only_dataset() -> EdtSegDataset:
    config = {
        "crop_size": [64, 64, 64],
        "sample_mode": "wrap",
        "cond_percent": [0.35, 0.35],
        "use_triplet_wrap_displacement": True,
        "use_dense_displacement": True,
        "use_extrapolation": False,
        "use_other_wrap_cond": False,
        "use_sdt": False,
        "use_heatmap_targets": False,
        "use_segmentation": False,
        "validate_result_tensors": False,
        "triplet_warn_missing_overlap_masks": False,
        "verbose": False,
    }
    setdefault_rowcol_cond_dataset_config(config)
    validate_rowcol_cond_dataset_config(config)

    patch_metadata = {
        "patches": [],
        "sample_index": [],
        "triplet_neighbor_lookup": {},
        "triplet_lookup_stats": {},
        "triplet_overlap_filter_stats": {},
        "triplet_overlap_kept_indices": tuple(),
        "cond_percent": [0.35, 0.35],
        "sample_mode": "wrap",
        "use_triplet_wrap_displacement": True,
    }
    return EdtSegDataset(config=config, apply_augmentation=False, patch_metadata=patch_metadata)


def _build_chunk(chunk_id, wrap_specs, segments_by_name):
    wraps = []
    segments = []
    for seg_idx, spec in enumerate(wrap_specs):
        segment_name = spec["segment_name"]
        segment = segments_by_name[segment_name]
        segments.append(segment)
        wraps.append(
            {
                "segment": segment,
                "bbox_2d": projected_bbox_2d(spec["cutout_name"], segment_name),
                "wrap_id": seg_idx,
                "segment_idx": seg_idx,
            }
        )

    return ChunkPatch(
        chunk_id=chunk_id,
        volume=np.zeros((1, 1, 1), dtype=np.float32),
        scale=0,
        world_bbox=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        wraps=wraps,
        segments=segments,
    )


def test_dataset_triplet_overlap_filter_drops_overlap_chunks_keeps_no_overlap_chunks(monkeypatch):
    parent_masks = detect_parent_overlap_masks()
    segments_by_name = load_parent_tifxyz_segments()
    dataset = _build_filter_only_dataset()
    cutouts = cutout_name_map()

    patches = [
        _build_chunk(
            (0, 0, 0),
            [
                {"segment_name": "w018_20260130192417321", "cutout_name": cutouts["w018_w019_overlap"]},
                {"segment_name": "w019_20260203030108447", "cutout_name": cutouts["w018_w019_overlap"]},
            ],
            segments_by_name,
        ),
        _build_chunk(
            (1, 0, 0),
            [
                {"segment_name": "w020_20260219135139420", "cutout_name": cutouts["w019_w020_overlap"]},
                {"segment_name": "w019_20260203030108447", "cutout_name": cutouts["w019_w020_overlap"]},
            ],
            segments_by_name,
        ),
        _build_chunk(
            (2, 0, 0),
            [
                {"segment_name": "w018_20260130192417321", "cutout_name": cutouts["w018_no_overlap"]},
                {"segment_name": "w019_20260203030108447", "cutout_name": cutouts["w019_no_overlap"]},
            ],
            segments_by_name,
        ),
    ]

    original_imread = rowcol_dataset_module.tifffile.imread
    mask_image_by_path = {}
    for parent_name, mask in parent_masks.items():
        path = (PHERC0139_ROOT / parent_name / "overlap_mask.tif").resolve()
        mask_image_by_path[str(path)] = mask.astype(np.uint8) * 255

    def _imread_override(path, *args, **kwargs):
        key = str(Path(path).resolve())
        if key in mask_image_by_path:
            return mask_image_by_path[key]
        return original_imread(path, *args, **kwargs)

    monkeypatch.setattr(rowcol_dataset_module.tifffile, "imread", _imread_override)

    kept = dataset._filter_triplet_overlap_chunks(patches)
    kept_chunk_ids = [tuple(p.chunk_id) for p in kept]
    assert kept_chunk_ids == [(2, 0, 0)]

    assert dataset._triplet_overlap_filter_stats == {
        "chunks_total": 3,
        "chunks_dropped_overlap": 2,
        "chunks_kept": 1,
        "missing_masks": 0,
    }
    assert dataset._triplet_overlap_kept_indices == (2,)
