"""Tests for categorize_zarr_files' resolution parsing.

Covers a real bug: the volume and segment regexes hardcoded exactly two
decimal places for the resolution (``\\d+\\.\\d{2}``). Of the resolutions
actually present in the public open-data bucket, only 3.24um and 7.91um
have two decimals -- 1.129um, 2.399um, 9.9um and 45.532um all have a
different number, and were silently dropped from the returned catalog with
no warning, leaving a local scan of a real data directory quietly
incomplete.
"""
import pytest

from vesuvius.data.paths.local import categorize_zarr_files


# Resolutions confirmed present in the public bucket's published paths.
REAL_RESOLUTIONS = ["1.129", "2.399", "3.24", "7.91", "9.9", "45.532"]


def _tree_for(resolutions):
    tree = {}
    for res in resolutions:
        tree[f"Scroll1/volumes/54keV_{res}um.zarr"] = None
        tree[f"Scroll1/segments/54keV_{res}um/seg_{res}.zarr"] = None
    return tree


def _collect(result):
    """Flatten the nested result into (resolutions_with_volume, resolutions_with_segment)."""
    vols, segs = set(), set()
    for _scroll, intensities in result.items():
        for _intensity, by_res in intensities.items():
            for res, entry in by_res.items():
                if entry.get("volume"):
                    vols.add(res)
                if entry.get("segments"):
                    segs.add(res)
    return vols, segs


@pytest.mark.parametrize("resolution", REAL_RESOLUTIONS)
def test_each_real_resolution_is_catalogued(resolution):
    """Every resolution that actually appears in published data must be
    parsed, regardless of how many decimal places it has."""
    result = categorize_zarr_files(_tree_for([resolution]), "/base")
    vols, segs = _collect(result)

    assert resolution in vols, (
        f"volume at resolution {resolution}um was silently skipped -- "
        f"the resolution pattern must not assume a fixed number of decimals"
    )
    assert resolution in segs, (
        f"segment at resolution {resolution}um was silently skipped"
    )


def test_all_real_resolutions_catalogued_together():
    """The full realistic mix must come through completely, not just the
    two-decimal subset."""
    result = categorize_zarr_files(_tree_for(REAL_RESOLUTIONS), "/base")
    vols, segs = _collect(result)

    assert vols == set(REAL_RESOLUTIONS), (
        f"missing volume resolutions: {set(REAL_RESOLUTIONS) - vols}"
    )
    assert segs == set(REAL_RESOLUTIONS), (
        f"missing segment resolutions: {set(REAL_RESOLUTIONS) - segs}"
    )


def test_two_decimal_resolutions_still_work():
    """No-regression check for the resolutions that already parsed."""
    result = categorize_zarr_files(_tree_for(["3.24", "7.91"]), "/base")
    vols, segs = _collect(result)
    assert vols == {"3.24", "7.91"}
    assert segs == {"3.24", "7.91"}


def test_integer_resolution_is_catalogued():
    """A resolution with no decimal point at all must not be skipped."""
    result = categorize_zarr_files(_tree_for(["20"]), "/base")
    vols, _segs = _collect(result)
    assert "20" in vols


def test_volume_path_and_segment_id_are_preserved():
    """Beyond just matching, the captured fields must still be correct."""
    tree = {"Scroll1/volumes/54keV_1.129um.zarr": None,
            "Scroll1/segments/54keV_1.129um/my_segment.zarr": None}
    result = categorize_zarr_files(tree, "/base")

    assert "1" in result
    assert "54" in result["1"]
    entry = result["1"]["54"]["1.129"]
    assert entry["volume"].endswith("Scroll1/volumes/54keV_1.129um.zarr")
    assert "my_segment" in entry["segments"]
    assert entry["segments"]["my_segment"].endswith(
        "Scroll1/segments/54keV_1.129um/my_segment.zarr"
    )
