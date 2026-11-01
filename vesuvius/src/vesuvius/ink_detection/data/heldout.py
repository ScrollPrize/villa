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

from collections.abc import Sequence
from dataclasses import replace

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.types import Patch, Segment
from vesuvius.ink_detection.volume_io import normalized_source_path


def has_heldout_validation(config: InkDataConfig) -> bool:
    return config.discovery_mode != "unlabeled" and any(
        source.split == "validation" for source in config.datasets
    )


def partition_heldout(patches: Sequence[Patch]) -> list[Patch]:
    """Score held-out datasets whole and drop validation windows inside training data.

    Training patches of training datasets come back unchanged and in their original
    order, so the training side of a run is the same as without the held-out datasets.
    A held-out dataset's training windows become validation patches scored on its
    supervision mask excluding validation voxels; its validation-mask windows stay
    validation patches.
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
                    exclude_validation_mask=True,
                )
            )
        elif not patch.is_validation:
            partitioned.append(patch)
    return partitioned


def assert_heldout_isolated(
    training: Sequence[Patch],
    validation: Sequence[Patch],
    config: InkDataConfig,
    segments: Sequence[Segment],
) -> dict:
    """Refuse a held-out split unless validation is spatially isolated from training.

    Held-out datasets must name a scroll that no training dataset uses, so the two
    sides are different physical objects; they must share no image volume, segment
    directory or label asset (resolving local path aliases). Every held-out segment
    must contribute validation patches, so a stale patch cache cannot drop one
    quietly. Returns the counts that were checked.
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
        return {normalized_source_path(value) for value in found}

    shared_assets = sorted(assets(training) & assets(validation))
    if shared_assets:
        raise ValueError(
            "held-out validation shares image volumes, segment directories or label "
            f"assets with training: {shared_assets[:3]}"
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
    }
