"""Resuming from a checkpoint continues the training sample stream instead of replaying it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader
from torch.utils.data import DataLoader

from vesuvius.ink_detection.config import (
    InkDataConfig,
    TrainingConfig,
    resolve_training_mapping,
)
from vesuvius.ink_detection.training.samplers import (
    build_sampling_policy,
    resume_position,
    resumed_stream_steps,
    sampling_stream_key,
    training_batch_sampler,
)
from vesuvius.ink_detection.types import Patch, Segment

CONFIGS = Path(__file__).resolve().parents[2] / "src/vesuvius/ink_detection/configs"
STRATEGIES = ["uniform", "scroll_segment_balanced", "fixed_scroll_prior_stratified"]
FIXED = "fixed_scroll_prior_stratified"


@pytest.fixture(autouse=True, scope="module")
def _accelerator_state():
    # DataLoaderShard synchronizes the shuffling generator through the
    # accelerator state, as in training; one CPU process here.
    Accelerator(cpu=True)


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


def _policy(tmp_path: Path, strategy: str, counts=(2, 2, 4)):
    """Build the policy the way train.py does on every (re)start: from the config seed."""
    config = _config(tmp_path, strategy)
    patches = []
    for (dataset_idx, relpath), count in zip(((0, "a1"), (0, "a2"), (1, "b1")), counts):
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


def _main_loader(policy, dataset) -> DataLoader:
    """The training DataLoader exactly as main builds it."""
    if policy.batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=policy.batch_sampler)
    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=policy.shuffle,
        sampler=policy.sampler,
        generator=policy.generator,
    )


def _new_loader(policy, dataset):
    batch_sampler = training_batch_sampler(policy, dataset, batch_size=4)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, generator=policy.generator)
    return loader, batch_sampler


def _prepared(loader: DataLoader, num_processes: int, rank: int):
    """Shard the loader the way accelerator.prepare does for one of num_processes ranks."""
    return prepare_data_loader(
        loader,
        num_processes=num_processes,
        process_index=rank,
        put_on_device=False,
        rng_types=["generator"],
    )


def _steps(loader, steps: int) -> list[list[int]]:
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


@pytest.mark.parametrize("num_processes", [1, 2])
@pytest.mark.parametrize(
    "strategy,counts",
    [
        ("uniform", (2, 2, 3)),  # 7 samples: an incomplete last batch
        ("uniform", (2, 2, 4)),
        ("scroll_segment_balanced", (2, 2, 3)),
        (FIXED, (3, 3, 6)),  # 3 batches per epoch: one is unpaired on 2 ranks
        (FIXED, (2, 2, 4)),
    ],
)
def test_fresh_runs_shard_and_draw_exactly_like_main(tmp_path, strategy, counts, num_processes):
    dataset = list(range(sum(counts)))
    for rank in range(num_processes):
        main = _prepared(_main_loader(_policy(tmp_path, strategy, counts), dataset), num_processes, rank)
        new, _ = _new_loader(_policy(tmp_path, strategy, counts), dataset)
        new = _prepared(new, num_processes, rank)
        assert len(new) == len(main)
        main_steps, new_steps = _steps(main, 9), _steps(new, 9)
        assert new_steps == main_steps
        if num_processes > 1:
            # Every rank gets full batches, so the preview gather sees equal shapes.
            assert all(len(batch) == 4 for batch in new_steps)


@pytest.mark.parametrize("num_processes", [1, 2, 3])
@pytest.mark.parametrize("counts", [(3, 3, 6), (2, 2, 4), (5, 5, 10)])
def test_resumed_fixed_prior_run_continues_the_uninterrupted_stream(tmp_path, counts, num_processes):
    if sum(counts) // 4 < num_processes:
        pytest.skip("fewer fixed-prior batches per epoch than processes: no full step")
    dataset = list(range(sum(counts)))
    total, after = 12, 5
    uninterrupted = []
    for rank in range(num_processes):
        loader, _ = _new_loader(_policy(tmp_path, FIXED, counts), dataset)
        uninterrupted.append(_steps(_prepared(loader, num_processes, rank), total + after))
    for completed in range(total + 1):
        for rank in range(num_processes):
            loader, batch_sampler = _new_loader(_policy(tmp_path, FIXED, counts), dataset)
            prepared = _prepared(loader, num_processes, rank)
            batch_sampler.fast_forward(*resume_position(completed, len(prepared), num_processes))
            resumed = _steps(prepared, after)
            assert resumed == uninterrupted[rank][completed : completed + after], (completed, rank)


def test_two_ranks_three_batch_epoch_skips_the_dropped_tail(tmp_path):
    """Review case: 2 ranks, 3 fixed-prior batches per epoch, the unpaired one is dropped."""
    counts = (3, 3, 6)
    dataset = list(range(12))
    raw = []
    inner = _policy(tmp_path, FIXED, counts).batch_sampler
    for _ in range(2):
        raw.extend(list(inner))
    loader, batch_sampler = _new_loader(_policy(tmp_path, FIXED, counts), dataset)
    prepared = _prepared(loader, 2, 0)
    assert len(prepared) == 1
    assert resume_position(1, len(prepared), 2) == (1, 0)
    batch_sampler.fast_forward(*resume_position(1, len(prepared), 2))
    # Step 2 of rank 0 is raw batch 4 (the first of the second pass), not raw batch 3.
    assert sorted(_steps(prepared, 1)[0]) == sorted(raw[3])


@pytest.mark.parametrize("strategy", ["uniform", "scroll_segment_balanced"])
def test_resumed_generator_runs_do_not_replay_the_first_batches(tmp_path, strategy):
    dataset = list(range(8))
    first, _ = _new_loader(_policy(tmp_path, strategy), dataset)
    first_run = _steps(first, 6)
    loader, batch_sampler = _new_loader(_policy(tmp_path, strategy), dataset)
    batch_sampler.fast_forward(*resume_position(3, len(loader), 1))
    assert _steps(loader, 6) != first_run


def test_audit_reports_skipped_draws_apart_from_loader_requests(tmp_path):
    dataset = list(range(12))
    policy = _policy(tmp_path, FIXED, (3, 3, 6))
    loader, batch_sampler = _new_loader(policy, dataset)
    batch_sampler.fast_forward(1, 1)
    audit = policy.batch_sampler.observed_audit()
    assert audit["batches_yielded_to_dataloader"] == 0
    assert audit["observed_by_scroll"] == {}
    assert audit["fast_forwarded"]["batches"] == 4
    assert audit["fast_forwarded"]["by_scroll"] == {"A": 8, "B": 8}
    _steps(loader, 2)
    audit = policy.batch_sampler.observed_audit()
    assert audit["batches_yielded_to_dataloader"] == 2
    assert audit["observed_by_scroll"] == {"A": 4, "B": 4}
    assert audit["fast_forwarded"]["batches"] == 4


def _recipe(name: str, **changes) -> dict:
    authored = json.loads((CONFIGS / f"{name}.json").read_text(encoding="utf-8"))
    authored.update(changes)
    return TrainingConfig.from_mapping(resolve_training_mapping(authored)).to_mapping()


def _key(mapping, *, num_processes=1, train_samples=1000, audit=None):
    return sampling_stream_key(
        mapping,
        num_processes=num_processes,
        train_samples=train_samples,
        policy_audit=audit or {"strategy": "uniform", "patches": train_samples},
    )


def test_dino_phase_transitions_start_new_streams():
    teacher, v1, v2 = (_recipe(f"dino_guided_{p}") for p in ("teacher", "v1", "v2"))
    v3, v3_fullsup = _recipe("dino_guided_v3"), _recipe("dino_guided_v3_fullsup")
    keys = [_key(m) for m in (teacher, v1, v2, v3, v3_fullsup)]
    assert len(set(keys)) == len(keys)
    # The teacher writes ckpt_060001 from its stream; v1 restores it and starts afresh.
    assert resumed_stream_steps({"key": keys[0], "start_step": 0}, keys[1], 60001) == (60001, 0)
    # v1 writes ckpt_063001 from the stream it started at 60001; v2 starts afresh.
    assert resumed_stream_steps({"key": keys[1], "start_step": 60001}, keys[2], 63001) == (63001, 0)
    # Both v3 recipes restore v2's ckpt_077001 and start afresh.
    for key in keys[3:]:
        assert resumed_stream_steps({"key": keys[2], "start_step": 63001}, key, 77001) == (77001, 0)


def test_resuming_the_same_recipe_continues_its_stream():
    v2 = _recipe("dino_guided_v2")
    resumed = _recipe(
        "dino_guided_v2",
        checkpoint="/path/to/runs/dino-guided-v2/ckpt_070001.pth",
        num_iterations=90001,
        save_iterations=[90001],
        out_dir="/path/to/runs/dino-guided-v2-resumed",
        dataloader_workers=2,
        description="resumed after a crash",
    )
    assert _key(resumed) == _key(v2)
    # Within the v2 stream (started at 63001), a resume at 70001 skips the 7000 steps taken in it.
    assert resumed_stream_steps({"key": _key(v2), "start_step": 63001}, _key(resumed), 70001) == (63001, 7000)


def test_stream_key_tracks_process_count_and_data():
    v2 = _recipe("dino_guided_v2")
    assert _key(v2, num_processes=2) != _key(v2)
    assert _key(v2, train_samples=999) != _key(v2)
    assert _key(v2, audit={"strategy": "uniform", "patches": 1000, "x": 1}) != _key(v2)


@pytest.mark.parametrize(
    "record",
    [None, {}, {"start_step": 10}, {"key": "other", "start_step": 0}, "not a mapping"],
)
def test_checkpoints_without_a_matching_stream_start_a_new_one(record):
    assert resumed_stream_steps(record, "k", 500) == (500, 0)


def test_weights_only_and_inconsistent_records_skip_nothing():
    assert resumed_stream_steps({"key": "k", "start_step": 0}, "k", 0) == (0, 0)
    assert resumed_stream_steps({"key": "k", "start_step": 600}, "k", 500) == (500, 0)
    assert resumed_stream_steps({"key": "k", "start_step": "x"}, "k", 500) == (500, 0)


def test_resume_position_counts_whole_loader_epochs_then_per_rank_batches():
    assert resume_position(0, 5, 2) == (0, 0)
    assert resume_position(4, 5, 2) == (0, 8)
    assert resume_position(5, 5, 2) == (1, 0)
    assert resume_position(12, 5, 3) == (2, 6)
    with pytest.raises(ValueError):
        resume_position(3, 0, 1)
