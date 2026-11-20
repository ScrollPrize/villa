"""Resuming from a checkpoint continues the training sample stream instead of replaying it."""

from __future__ import annotations

from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.training.samplers import (
    build_sampling_policy,
    training_batch_sampler,
)
from vesuvius.ink_detection.types import Patch, Segment


def _config(tmp_path: Path, strategy: str) -> InkDataConfig:
    return InkDataConfig.from_mapping(
        {
            "mode": "flat",
            "patch_size": [1, 2, 2],
            "patch_overlap": 0.25,
            "patch_min_labeled_coverage": 0.0,
            "datasets": [
                {
                    "segments_path": str(tmp_path / "a"),
                    "volume_scale": 0,
                    "sampling_scroll": "A",
                    "sampling_physical_segment_keys": {"a1": "A:1", "a2": "A:2"},
                    "sampling_representation_keys": {"a1": "ra1", "a2": "ra2"},
                },
                {
                    "segments_path": str(tmp_path / "b"),
                    "volume_scale": 0,
                    "sampling_scroll": "B",
                    "sampling_physical_segment_keys": {"b1": "B:1"},
                    "sampling_representation_keys": {"b1": "rb1"},
                },
            ],
            "seed": 42,
            "sampling_strategy": strategy,
            "fixed_scroll_prior": {"seed": 42, "target_batch_counts": {"A": 2, "B": 2}},
        }
    )


def _policy(tmp_path: Path, strategy: str):
    """Build the policy the way train.py does on every (re)start: from the config seed."""
    config = _config(tmp_path, strategy)
    patches = []
    for dataset_idx, relpath, count in ((0, "a1", 2), (0, "a2", 2), (1, "b1", 4)):
        segment = Segment(
            data_config=config,
            source=config.datasets[dataset_idx],
            dataset_idx=dataset_idx,
            segment_relpath=relpath,
            segment_dir=tmp_path / relpath,
            segment_name=relpath,
            image_volume="unused",
        )
        patches.extend(Patch(segment=segment, bbox=(0, 0, 0, 1, 2, 2)) for _ in range(count))
    return build_sampling_policy(patches, config, batch_size=4)


def _steps(loader: DataLoader, steps: int) -> list[list[int]]:
    """What the training loop sees: one batch per step, re-entering the loader at epoch end."""
    iterator = iter(loader)
    seen = []
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        seen.append(batch.tolist())
    return seen


def _loader(policy, *, resume_after: int = 0) -> DataLoader:
    batch_sampler = training_batch_sampler(policy, 8, batch_size=4)
    batch_sampler.fast_forward(resume_after)
    return DataLoader(list(range(8)), batch_sampler=batch_sampler, generator=policy.generator)


STRATEGIES = ["uniform", "scroll_segment_balanced", "fixed_scroll_prior_stratified"]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_fresh_runs_draw_the_same_batches_as_before(tmp_path, strategy):
    old = _policy(tmp_path, strategy)
    if old.batch_sampler is not None:
        before = DataLoader(list(range(8)), batch_sampler=old.batch_sampler)
    else:
        before = DataLoader(
            list(range(8)),
            batch_size=4,
            shuffle=old.shuffle,
            sampler=old.sampler,
            generator=old.generator,
        )
    assert _steps(_loader(_policy(tmp_path, strategy)), 7) == _steps(before, 7)


def test_resumed_fixed_prior_run_continues_the_uninterrupted_stream(tmp_path):
    strategy = "fixed_scroll_prior_stratified"
    uninterrupted = _steps(_loader(_policy(tmp_path, strategy)), 7)
    # What a resume did before: rebuild from the seed and start again at batch 0.
    replayed = _steps(_loader(_policy(tmp_path, strategy)), 4)
    assert replayed == uninterrupted[:4]
    assert replayed != uninterrupted[3:]
    # Resume after 3 trained steps (more than one 2-batch epoch): exact continuation.
    assert _steps(_loader(_policy(tmp_path, strategy), resume_after=3), 4) == uninterrupted[3:]


@pytest.mark.parametrize("strategy", ["uniform", "scroll_segment_balanced"])
def test_resumed_generator_runs_do_not_replay_the_first_batches(tmp_path, strategy):
    first_run = _steps(_loader(_policy(tmp_path, strategy)), 6)
    resumed = _steps(_loader(_policy(tmp_path, strategy), resume_after=3), 6)
    assert resumed != first_run
