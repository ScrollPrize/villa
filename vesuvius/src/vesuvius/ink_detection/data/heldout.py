"""Validate on whole held-out datasets, and refuse a split whose crops can touch.

The default finder takes validation windows from a validation mask inside the
training segments. Labels never mix, because training supervision excludes the
validation voxels, but a validation crop and a training crop can cover the same
image voxels, so validation measures interpolation inside a segment the model
was trained on. A dataset marked ``"split": "validation"`` is held out whole:
every window it would have trained on is scored instead, nothing of it reaches
training, and validation reads nothing else.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.types import Patch, Segment


def has_heldout_validation(config: InkDataConfig) -> bool:
    return config.discovery_mode != "unlabeled" and any(
        source.split == "validation" for source in config.datasets
    )


def partition_heldout(patches: Sequence[Patch]) -> list[Patch]:
    """Score held-out datasets whole and drop validation windows inside training data.

    Training patches of training datasets come back unchanged and in their original
    order, so the training side of a run is the same as without the held-out datasets.
    A held-out dataset's training windows become validation patches scored on its
    supervision mask; its validation-mask windows stay validation patches.
    """
    partitioned: list[Patch] = []
    for patch in patches:
        if patch.segment.source.split == "validation":
            partitioned.append(
                patch
                if patch.is_validation
                else replace(
                    patch,
                    is_validation=True,
                    supervision_mask_override=patch.segment.supervision_mask,
                )
            )
        elif not patch.is_validation:
            partitioned.append(patch)
    return partitioned


def _volume_key(patch: Patch) -> tuple[str, int]:
    return str(patch.image_volume), int(patch.segment.scale)


def crop_overlap_pairs(
    training: Sequence[Patch], validation: Sequence[Patch]
) -> int:
    """Count training/validation crop pairs that share a voxel of one image volume.

    Boxes are half-open ZYX bounds, so crops that only touch along a face do not count.
    """
    boxes: dict[tuple[str, int], tuple[list, list]] = defaultdict(lambda: ([], []))
    for patch in training:
        boxes[_volume_key(patch)][0].append(tuple(int(v) for v in patch.bbox))
    for patch in validation:
        boxes[_volume_key(patch)][1].append(tuple(int(v) for v in patch.bbox))
    pairs = 0
    for train_boxes, val_boxes in boxes.values():
        if not train_boxes or not val_boxes:
            continue
        train_boxes.sort(key=lambda box: box[1])
        starts = [box[1] for box in train_boxes]
        tallest = max(box[4] - box[1] for box in train_boxes)
        for vz0, vy0, vx0, vz1, vy1, vx1 in val_boxes:
            lo = bisect_right(starts, vy0 - tallest)
            hi = bisect_left(starts, vy1)
            for tz0, ty0, tx0, tz1, ty1, tx1 in train_boxes[lo:hi]:
                if (
                    ty1 > vy0
                    and tx0 < vx1
                    and tx1 > vx0
                    and tz0 < vz1
                    and tz1 > vz0
                ):
                    pairs += 1
    return pairs


def assert_heldout_isolated(
    training: Sequence[Patch],
    validation: Sequence[Patch],
    config: InkDataConfig,
    segments: Sequence[Segment],
) -> dict:
    """Refuse a held-out split unless validation is spatially isolated from training.

    Held-out datasets must name a scroll that no training dataset uses, so the two
    sides are different physical objects; they must share no image volume, segment
    directory or label asset; and no validation crop may share a voxel with a
    training crop. Every held-out segment must contribute validation patches, so a
    stale patch cache cannot drop one quietly. Returns the counts that were checked.
    """
    sources = config.datasets
    for index, source in enumerate(sources):
        if not source.sampling_scroll:
            raise ValueError(
                f"datasets[{index}] must define sampling_scroll when any dataset is "
                "held out for validation, so the held-out scroll can be checked"
            )
    train_scrolls = {s.sampling_scroll for s in sources if s.split == "train"}
    heldout_scrolls = {s.sampling_scroll for s in sources if s.split == "validation"}
    shared_scrolls = sorted(train_scrolls & heldout_scrolls)
    if shared_scrolls:
        raise ValueError(
            "held-out validation datasets must come from scrolls that no training "
            f"dataset uses; shared: {shared_scrolls}"
        )
    misplaced = [p for p in training if p.segment.source.split != "train"]
    misplaced += [p for p in validation if p.segment.source.split != "validation"]
    if misplaced:
        raise ValueError(
            f"{len(misplaced)} patches sit on the wrong side of the held-out split, "
            f"first: {misplaced[0].segment.segment_relpath!r} {misplaced[0].bbox!r}"
        )
    heldout_segments = {
        segment.cache_key for segment in segments if segment.source.split == "validation"
    }
    scored = {patch.segment.cache_key for patch in validation}
    missing = sorted(key[1] for key in heldout_segments - scored)
    if not validation or missing:
        raise ValueError(
            "held-out validation found no patches"
            + (f" for {missing}" if missing else "")
        )

    def assets(patches: Sequence[Patch]) -> set[str]:
        found: set[str] = set()
        for patch in patches:
            segment = patch.segment
            found.update(
                str(value)
                for value in (
                    segment.image_volume,
                    segment.segment_dir,
                    segment.inklabels,
                    segment.supervision_mask,
                    segment.validation_mask,
                )
                if value is not None
            )
        return found

    shared_assets = sorted(assets(training) & assets(validation))
    if shared_assets:
        raise ValueError(
            "held-out validation shares image volumes, segment directories or label "
            f"assets with training: {shared_assets[:3]}"
        )
    overlaps = crop_overlap_pairs(training, validation)
    if overlaps:
        raise ValueError(
            f"{overlaps} validation crops share voxels with training crops"
        )
    return {
        "training_patches": len(training),
        "validation_patches": len(validation),
        "training_scrolls": sorted(train_scrolls),
        "validation_scrolls": sorted(heldout_scrolls),
        "validation_segments": sorted(
            {f"{p.segment.dataset_idx}:{p.segment.segment_relpath}" for p in validation}
        ),
        "shared_assets": 0,
        "overlapping_crop_pairs": overlaps,
    }
