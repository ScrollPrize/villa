"""Unit tests for the catalog provenance checker (issue #1730). No network."""

from __future__ import annotations

import gzip
import hashlib
import json

from vesuvius.data.catalog_provenance_check import (
    find_future_scan_segments,
    load_metadata,
    parse_date,
)


def test_parse_date_handles_catalog_forms():
    # Trailing Z, fractional seconds.
    assert parse_date("2025-05-02T18:34:21.033620Z").date().isoformat() == "2025-05-02"
    # Z without fraction.
    assert parse_date("2025-05-21T15:12:20Z").date().isoformat() == "2025-05-21"
    # Numeric offset is normalised to UTC (03:10+02:00 -> 01:10Z, same day).
    assert parse_date("2025-05-09T03:10:39.898839+02:00").date().isoformat() == "2025-05-09"
    # Malformed offset-plus-Z form seen in the catalog must not be dropped.
    assert parse_date("2025-01-01T00:00:00+00:00Z") is not None
    # Naive timestamps are assumed UTC.
    assert parse_date("2025-01-08T00:00:00").date().isoformat() == "2025-01-08"


def test_parse_date_returns_none_for_bad_input():
    assert parse_date(None) is None
    assert parse_date("") is None
    assert parse_date("not a date") is None
    assert parse_date(12345) is None


def _sample(segments, volumes, scans):
    return {"samples": {"PHercX": {"segments": segments, "volumes": volumes, "scans": scans}}}


def test_flags_segment_whose_scan_is_newer():
    # Volume/scan ids are 14-digit timestamps, as in the real catalog.
    meta = _sample(
        segments={
            "seg_bad": {
                "original_volume_id": "20250521151220",
                "creation": {"date": "2025-05-02T00:00:00Z"},
                "data": [{"origins": [{"path": "PHercX/...-volume-20250521151220.zarr"}]}],
            },
            "seg_ok": {
                "original_volume_id": "20250521151220",
                "creation": {"date": "2025-06-01T00:00:00Z"},
                "data": [],
            },
        },
        volumes={"20250521151220": {"scan_id": "20250509011039"}},
        scans={"20250509011039": {"creation": {"date": "2025-05-09T00:00:00Z"}}},
    )
    findings, total, comparable = find_future_scan_segments(meta)
    assert total == 2
    assert comparable == 2
    assert len(findings) == 1
    f = findings[0]
    assert f.segment_id == "seg_bad"
    assert f.sample == "PHercX"
    assert f.scan_id == "20250509011039"
    assert f.gap_days == 7
    # The declared id appears in seg_bad's own data paths -> pointer-consistent.
    assert f.declared_id_in_own_paths is True


def test_unresolvable_dates_are_not_comparable():
    meta = _sample(
        segments={
            "seg": {
                "original_volume_id": "missing",
                "creation": {"date": "2025-05-02T00:00:00Z"},
                "data": [],
            }
        },
        volumes={},  # original_volume_id does not resolve
        scans={},
    )
    findings, total, comparable = find_future_scan_segments(meta)
    assert total == 1
    assert comparable == 0
    assert findings == []


def test_revision_hashes_decompressed_bytes(tmp_path):
    # A plain .json copy and its gzip-encoded form of the same revision must
    # produce the same digest, so runs are comparable regardless of encoding.
    payload = json.dumps({"samples": {}}, sort_keys=True).encode("utf-8")
    expected = hashlib.sha256(payload).hexdigest()

    plain = tmp_path / "metadata.json"
    plain.write_bytes(payload)
    gzipped = tmp_path / "metadata.json.gz"
    gzipped.write_bytes(gzip.compress(payload))

    _, rev_plain = load_metadata(None, str(plain))
    _, rev_gz = load_metadata(None, str(gzipped))

    assert rev_plain.sha256 == expected
    assert rev_gz.sha256 == expected
    assert rev_plain.n_bytes == len(payload)
    assert expected in rev_plain.summary()


def test_declared_id_absent_from_paths_is_false():
    meta = _sample(
        segments={
            "seg": {
                "original_volume_id": "20250521151220",
                "creation": {"date": "2025-05-02T00:00:00Z"},
                "data": [{"origins": [{"path": "PHercX/...-volume-20250101000000.zarr"}]}],
            }
        },
        volumes={"20250521151220": {"scan_id": "20250509011039"}},
        scans={"20250509011039": {"creation": {"date": "2025-05-09T00:00:00Z"}}},
    )
    findings, _, _ = find_future_scan_segments(meta)
    assert len(findings) == 1
    assert findings[0].declared_id_in_own_paths is False
