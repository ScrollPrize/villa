from __future__ import annotations

import pytest

from vesuvius.scripts.fill_inner_outer_labels import group_by_output_chunk


def _chunks_written(slice_idx: int, chunk_z: int, scale_factor: int) -> set[int]:
    start = slice_idx * scale_factor
    return {z // chunk_z for z in range(start, start + scale_factor)}


@pytest.mark.parametrize(
    "start, stop, chunk_z, scale_factor",
    [(384, 768, 192, 1), (192, 384, 192, 2), (3, 97, 100, 8), (0, 10, 192, 128), (5, 300, 128, 4)],
)
def test_no_output_chunk_is_written_by_two_groups(start, stop, chunk_z, scale_factor) -> None:
    groups = group_by_output_chunk(list(range(start, stop)), lambda s: s, chunk_z, scale_factor)

    owner: dict[int, int] = {}
    for group_idx, group in enumerate(groups):
        for slice_idx in group:
            for chunk in _chunks_written(slice_idx, chunk_z, scale_factor):
                assert owner.setdefault(chunk, group_idx) == group_idx
    assert [s for group in groups for s in group] == list(range(start, stop))


def test_one_group_per_output_chunk_when_slices_align() -> None:
    groups = group_by_output_chunk(list(range(0, 400)), lambda s: s, 128)
    assert [len(group) for group in groups] == [128, 128, 128, 16]
