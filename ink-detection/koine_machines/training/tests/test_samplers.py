from dataclasses import dataclass

import pytest

from koine_machines.training.samplers import (
    FixedScrollPriorStratifiedBatchSampler,
    hierarchical_scroll_segment_weights,
)


@dataclass
class _Segment:
    dataset_idx: int
    segment_relpath: str


@dataclass
class _Patch:
    segment: _Segment


def _patches(dataset_idx, segment, count):
    return [_Patch(_Segment(dataset_idx, segment)) for _ in range(count)]


def test_hierarchical_weights_equalize_scroll_then_physical_segment():
    # Scroll A has two physical segments. a1 occurs in two representations,
    # while scroll B has one segment and far more raw windows.
    patches = [
        *_patches(0, "a1", 2),
        *_patches(1, "a1", 6),
        *_patches(0, "a2", 4),
        *_patches(2, "b1", 20),
    ]
    datasets = [
        {"sampling_scroll": "A"},
        {"sampling_scroll": "A"},
        {"sampling_scroll": "B"},
    ]

    weights, audit = hierarchical_scroll_segment_weights(patches, datasets)
    masses = {}
    for patch, weight in zip(patches, weights.tolist(), strict=True):
        key = (datasets[patch.segment.dataset_idx]["sampling_scroll"], patch.segment.segment_relpath)
        masses[key] = masses.get(key, 0.0) + weight

    assert sum(weight for (scroll, _), weight in masses.items() if scroll == "A") == pytest.approx(0.5)
    assert masses[("A", "a1")] == pytest.approx(0.25)
    assert masses[("A", "a2")] == pytest.approx(0.25)
    assert masses[("B", "b1")] == pytest.approx(0.5)
    assert audit["segments_per_scroll"] == {"A": 2, "B": 1}


def test_hierarchical_weights_require_explicit_scroll_identity():
    with pytest.raises(ValueError, match="sampling_scroll"):
        hierarchical_scroll_segment_weights(
            _patches(0, "a1", 1),
            [{}],
        )


def _fixed_prior_fixture():
    datasets = [
        {
            "sampling_scroll": "A",
            "sampling_physical_segment_keys": {
                "a1-public": "A:a1",
                "a2": "A:a2",
            },
            "sampling_representation_keys": {
                "a1-public": "public:a1",
                "a2": "public:a2",
            },
        },
        {
            "sampling_scroll": "A",
            "sampling_physical_segment_keys": {"a1-native": "A:a1"},
            "sampling_representation_keys": {"a1-native": "native:a1"},
        },
        {
            "sampling_scroll": "B",
            "sampling_physical_segment_keys": {"b1": "B:b1"},
            "sampling_representation_keys": {"b1": "public:b1"},
        },
        {
            "sampling_scroll": "C",
            "sampling_physical_segment_keys": {"c1": "C:c1"},
            "sampling_representation_keys": {"c1": "public:c1"},
        },
        {
            "sampling_scroll": "D",
            "sampling_physical_segment_keys": {"d1": "D:d1"},
            "sampling_representation_keys": {"d1": "public:d1"},
        },
    ]
    patches = [
        *_patches(0, "a1-public", 5),
        *_patches(0, "a2", 7),
        *_patches(1, "a1-native", 3),
        *_patches(2, "b1", 11),
        *_patches(3, "c1", 13),
        *_patches(4, "d1", 17),
    ]
    quotas = {"A": 3, "B": 2, "C": 2, "D": 1}
    return patches, datasets, quotas


def _sample_metadata(patches, datasets, sample_idx):
    patch = patches[sample_idx]
    dataset = datasets[patch.segment.dataset_idx]
    segment = patch.segment.segment_relpath
    return (
        dataset["sampling_scroll"],
        dataset["sampling_physical_segment_keys"][segment],
        dataset["sampling_representation_keys"][segment],
    )


def test_fixed_prior_enforces_every_batch_and_balances_hierarchy():
    patches, datasets, quotas = _fixed_prior_fixture()
    sampler = FixedScrollPriorStratifiedBatchSampler(
        patches,
        datasets,
        batch_quotas=quotas,
        batch_size=8,
        seed=42,
    )

    batches = [next(iter(sampler)) for _ in range(20)]
    physical_counts = {}
    representation_counts = {}
    for batch in batches:
        scroll_counts = {}
        for sample_idx in batch:
            scroll, physical, representation = _sample_metadata(
                patches, datasets, sample_idx
            )
            scroll_counts[scroll] = scroll_counts.get(scroll, 0) + 1
            physical_counts[physical] = physical_counts.get(physical, 0) + 1
            representation_counts[representation] = (
                representation_counts.get(representation, 0) + 1
            )
        assert scroll_counts == quotas

    assert abs(physical_counts["A:a1"] - physical_counts["A:a2"]) <= 1
    assert abs(
        representation_counts["public:a1"]
        - representation_counts["native:a1"]
    ) <= 1
    audit = sampler.observed_audit()
    assert audit["batches_yielded_to_dataloader"] == 20
    assert audit["observed_by_scroll"] == {
        scroll: count * 20 for scroll, count in quotas.items()
    }


def test_fixed_prior_patch_queues_do_not_repeat_before_recycling():
    patches, datasets, quotas = _fixed_prior_fixture()
    sampler = FixedScrollPriorStratifiedBatchSampler(
        patches,
        datasets,
        batch_quotas=quotas,
        batch_size=8,
        seed=42,
    )
    seen = []
    for batch in sampler:
        for sample_idx in batch:
            if _sample_metadata(patches, datasets, sample_idx)[2] == "native:a1":
                seen.append(sample_idx)
                if len(seen) == 3:
                    break
        if len(seen) == 3:
            break
    assert len(set(seen)) == 3


def test_fixed_prior_is_deterministic_from_seed():
    patches, datasets, quotas = _fixed_prior_fixture()

    def first_batches(seed):
        sampler = FixedScrollPriorStratifiedBatchSampler(
            patches,
            datasets,
            batch_quotas=quotas,
            batch_size=8,
            seed=seed,
        )
        iterator = iter(sampler)
        return [next(iterator) for _ in range(5)]

    assert first_batches(42) == first_batches(42)
    assert first_batches(42) != first_batches(43)


def test_fixed_prior_rejects_implicit_physical_segment_identity():
    patches, datasets, quotas = _fixed_prior_fixture()
    del datasets[0]["sampling_physical_segment_keys"]["a1-public"]
    with pytest.raises(ValueError, match="missing an explicit sampling mapping"):
        FixedScrollPriorStratifiedBatchSampler(
            patches,
            datasets,
            batch_quotas=quotas,
            batch_size=8,
            seed=42,
        )
