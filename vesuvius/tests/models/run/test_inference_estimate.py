"""Tests for --estimate, the pre-run cost report.

The arithmetic here is an exact count over two grids, not a timing, so it can be asserted
exactly rather than within a tolerance. That is the point of the feature: the expensive
number in a scroll-scale run is knowable before the run starts.
"""

import re
from types import SimpleNamespace

import pytest

from vesuvius.models.run import estimate
from vesuvius.models.run.inference import Inferer


# --- the chunk arithmetic ---------------------------------------------------------------

def test_patch_aligned_to_chunk_grid_touches_the_exact_span():
    # a 192 patch at the origin spans chunks 0 and 1 on a 128 grid, on every axis
    spans = estimate.chunk_ranges((0, 0, 0), (192, 192, 192), (128, 128, 128))
    assert [list(s) for s in spans] == [[0, 1], [0, 1], [0, 1]]


def test_patch_smaller_than_a_chunk_and_inside_it_touches_one():
    spans = estimate.chunk_ranges((10, 10, 10), (64, 64, 64), (128, 128, 128))
    assert [list(s) for s in spans] == [[0], [0], [0]]


def test_patch_straddling_a_boundary_touches_both_chunks():
    spans = estimate.chunk_ranges((120, 0, 0), (16, 16, 16), (128, 128, 128))
    assert list(spans[0]) == [0, 1]


def test_a_192_patch_on_a_128_grid_can_touch_27_chunks():
    # the worst case that motivates the whole feature: misaligned on all three axes
    spans = estimate.chunk_ranges((100, 100, 100), (192, 192, 192), (128, 128, 128))
    assert [len(list(s)) for s in spans] == [3, 3, 3]


# --- the simulation ---------------------------------------------------------------------

def test_no_cache_counts_every_touch_as_a_fetch():
    # Patch A at x=0 spans 0..191, so chunk columns 0 and 1: 2x2x2 = 8 touches.
    # Patch B at x=96 spans 96..287, so columns 0, 1 AND 2: 2x2x3 = 12 touches.
    # With no cache nothing is remembered between them, so it is the plain sum.
    positions = [(0, 0, 0), (0, 0, 96)]
    fetches, distinct = estimate.simulate(positions, (192,) * 3, (128,) * 3, cache_chunks=0)
    assert fetches == 8 + 12
    # they share columns 0 and 1, so the union is smaller than the sum
    assert distinct == 2 * 2 * 3


def test_the_half_overlap_step_is_what_makes_a_patch_span_three_columns():
    """Why the amplification exists at all, pinned as a test.

    192 and 128 do not align, so stepping by half a patch (96) walks the patch across a
    third chunk column even though the patch is only 1.5 chunks wide.
    """
    aligned = estimate.chunk_ranges((0, 0, 0), (192,) * 3, (128,) * 3)
    stepped = estimate.chunk_ranges((0, 0, 96), (192,) * 3, (128,) * 3)
    assert len(list(aligned[2])) == 2
    assert len(list(stepped[2])) == 3


def test_an_unbounded_cache_reaches_the_floor():
    positions = [(0, 0, 0), (0, 0, 96), (0, 96, 0)]
    fetches, distinct = estimate.simulate(
        positions, (192,) * 3, (128,) * 3, cache_chunks=10_000
    )
    assert fetches == distinct


def test_distinct_is_independent_of_cache_and_order():
    positions = [(z, y, 0) for z in (0, 96, 192) for y in (0, 96, 192)]
    seen = set()
    for order in estimate.ORDERS:
        for cache in (0, 4, 10_000):
            _, distinct = estimate.simulate(
                positions, (192,) * 3, (128,) * 3, cache_chunks=cache, order=order
            )
            seen.add(distinct)
    assert len(seen) == 1, f"distinct chunk count drifted with policy: {seen}"


def test_a_cache_never_costs_more_than_no_cache():
    positions = [(z, y, x) for z in (0, 96, 192) for y in (0, 96, 192) for x in (0, 96)]
    uncached, _ = estimate.simulate(positions, (192,) * 3, (128,) * 3, cache_chunks=0)
    cached, _ = estimate.simulate(positions, (192,) * 3, (128,) * 3, cache_chunks=8)
    assert cached <= uncached


def test_morton_is_not_worse_than_raster_at_the_same_cache():
    positions = [(z, y, x)
                 for z in range(0, 768, 96)
                 for y in range(0, 768, 96)
                 for x in range(0, 768, 96)]
    raster, _ = estimate.simulate(
        positions, (192,) * 3, (128,) * 3, cache_chunks=16, order="current"
    )
    morton, _ = estimate.simulate(
        positions, (192,) * 3, (128,) * 3, cache_chunks=16, order="morton"
    )
    assert morton <= raster


# --- ordering ---------------------------------------------------------------------------

def test_current_order_is_left_exactly_as_given():
    positions = [(192, 0, 0), (0, 0, 0), (96, 0, 0)]
    assert estimate.order_positions(positions, (128,) * 3, "current") == positions


def test_reordering_is_a_permutation_not_a_filter():
    positions = [(z, y, 0) for z in (0, 96, 192) for y in (0, 96, 192)]
    for order in estimate.ORDERS:
        out = estimate.order_positions(positions, (128,) * 3, order)
        assert sorted(out) == sorted(positions)


def test_unknown_order_is_refused():
    with pytest.raises(ValueError, match="unknown traversal order"):
        estimate.order_positions([(0, 0, 0)], (128,) * 3, "spiral")


def test_morton_key_interleaves_bits():
    assert estimate.morton_key((0, 0, 0)) == 0
    assert estimate.morton_key((0, 0, 1)) == 0b001
    assert estimate.morton_key((0, 1, 0)) == 0b010
    assert estimate.morton_key((1, 0, 0)) == 0b100
    assert estimate.morton_key((1, 1, 1)) == 0b111


# --- the plan ---------------------------------------------------------------------------

def _plan():
    positions = [(z, y, 0) for z in (0, 96, 192) for y in (0, 96, 192)]
    return estimate.build_plan(positions, (192,) * 3, (128,) * 3, itemsize=1)


def test_plan_reports_the_floor_and_the_chunk_size():
    plan = _plan()
    assert plan["chunk_bytes"] == 128 ** 3          # uint8, so 2 MiB
    assert plan["floor_bytes"] == plan["distinct_chunks"] * plan["chunk_bytes"]
    assert plan["n_patches"] == 9


def test_amplification_is_fetches_over_distinct():
    plan = _plan()
    for row in plan["rows"]:
        assert row["amplification"] == pytest.approx(row["fetches"] / plan["distinct_chunks"])
        assert row["bytes"] == row["fetches"] * plan["chunk_bytes"]


def test_the_uncached_row_is_the_worst_row():
    plan = _plan()
    uncached = max(r["fetches"] for r in plan["rows"] if r["cache_gib"] == 0)
    assert uncached == max(r["fetches"] for r in plan["rows"])


def test_report_names_the_compressor_only_when_there_is_one():
    plan = _plan()
    raw = "\n".join(estimate.format_plan(plan, 100.0, compressor=None))
    zstd = "\n".join(estimate.format_plan(plan, 100.0, compressor="zstd"))
    assert "uncompressed size" not in raw
    assert "uncompressed size" in zstd


def test_report_shows_every_policy_row():
    plan = _plan()
    text = "\n".join(estimate.format_plan(plan, 100.0))
    assert "current (villa)" in text
    assert "morton" in text
    assert "chunk-blocked" in text


# --- the wiring -------------------------------------------------------------------------

def test_estimate_reads_only_attributes_the_class_actually_sets():
    """Same guard as the resume signature, for the same reason.

    _run_signature shipped reading self.input_dir and self.disable_tta, neither of which
    exists, and every unit test passed because the fixtures invented them. The estimate
    path reads a fresh set of attributes, so it needs the same check.
    """
    import inspect

    class_src = inspect.getsource(Inferer)
    assigned = set(re.findall(r"self\.(\w+)\s*=", class_src))
    read = set()
    for method in (Inferer._report_estimate, Inferer._input_array_geometry):
        read |= set(re.findall(r"self\.(\w+)", inspect.getsource(method)))
    missing = sorted(read - assigned - {"_report_estimate", "_input_array_geometry"})
    assert not missing, f"the estimate path reads attributes the class never sets: {missing}"


def test_estimate_reports_patch_count_when_geometry_is_unreadable(capsys):
    """An unreadable store must not produce invented byte figures."""
    inf = Inferer.__new__(Inferer)
    inf.patch_start_coords_list = [(0, 0, 0), (0, 0, 96)]
    inf.dataset = SimpleNamespace(volume=None)
    inf._report_estimate()
    out = capsys.readouterr().out
    assert "2 patches" in out
    assert "not reported rather than guessed" in out
    assert "GiB" not in out
