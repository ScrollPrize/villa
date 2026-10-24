#!/usr/bin/env python3
"""Flag catalog segments whose declared original volume was scanned after them.

Issue #1730: for a number of published segments, ``original_volume_id`` points to
a volume whose scan has a ``creation.date`` LATER than the segment's own
``creation.date``. A segment cannot have been traced on a scan that did not exist
yet, so either ``original_volume_id`` or one of the dates is wrong as published.

This tool reads the public catalog ``metadata.json`` (no credentials needed) and,
for every segment, resolves

    segment.original_volume_id -> volumes[id].scan_id -> scans[id].creation.date

and compares that scan date with the segment's own ``creation.date``. It reports
every segment where the scan is newer than the segment, writes an optional CSV,
and exits non-zero when any are found (so it can gate CI or a data-publish step).

To help tell a stale *pointer* from a real mis-assignment, each row also reports
whether the declared ``original_volume_id`` actually appears among the segment's
own data paths (``...-volume-<id>...``). When it does, the field reads like a
per-sample pointer that was kept current (the common case for PHerc0139 and
PHerc1447); when it does not, the segment genuinely points elsewhere. The id test
alone cannot prove which volume a mesh was traced on (see #1730 for the PHerc1447
z-extent overrun evidence), so this is reported as context, not a verdict.

Usage::

    python -m vesuvius.data.catalog_provenance_check
    python -m vesuvius.data.catalog_provenance_check --metadata-file meta.json --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

DEFAULT_METADATA_URL = (
    "https://vesuvius-challenge-open-data.s3.amazonaws.com/metadata.json"
)

# Volume ids are 14-digit timestamps; data paths spell them as "...-volume-<id>".
_VOLUME_ID_IN_PATH = re.compile(r"volume-(\d{14})")


@dataclass(frozen=True)
class Finding:
    sample: str
    segment_id: str
    segment_date: str
    original_volume_id: str
    scan_id: str
    scan_date: str
    gap_days: int
    declared_id_in_own_paths: bool


def parse_date(value: Any) -> datetime | None:
    """Parse a catalog ISO date tolerantly, returning an aware UTC datetime.

    Handles a trailing ``Z``, a numeric offset, and the malformed ``+00:00Z``
    form seen in the catalog. Naive timestamps are assumed to be UTC. Returns
    ``None`` for anything unparseable rather than raising, so one bad record
    cannot hide the rest.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    # "+00:00Z" carries both an offset and a Z; drop the redundant trailing Z.
    if text.endswith("Z") and ("+" in text[10:] or "-" in text[10:]):
        text = text[:-1]
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_metadata(url: str | None, path: str | None) -> Mapping[str, Any]:
    """Load the catalog JSON from a local file or the public URL (gzip-aware)."""
    if path:
        with open(path, "rb") as handle:
            raw = handle.read()
    else:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            raw = response.read()
    # The published object is gzip-encoded; a local copy may already be plain.
    try:
        raw = gzip.decompress(raw)
    except (OSError, gzip.BadGzipFile):
        pass
    return json.loads(raw.decode("utf-8"))


def _creation_date(entry: Any) -> datetime | None:
    if not isinstance(entry, Mapping):
        return None
    creation = entry.get("creation")
    if not isinstance(creation, Mapping):
        return None
    return parse_date(creation.get("date"))


def find_future_scan_segments(metadata: Mapping[str, Any]) -> tuple[list[Finding], int, int]:
    """Return (findings, total_segments, comparable_segments).

    ``comparable_segments`` counts segments where both the segment date and the
    resolved scan date parsed, so the caller can report how many were checked.
    """
    findings: list[Finding] = []
    total = 0
    comparable = 0
    samples = metadata.get("samples", {})
    if not isinstance(samples, Mapping):
        return findings, total, comparable

    for sample_name, sample in samples.items():
        if not isinstance(sample, Mapping):
            continue
        segments = sample.get("segments", {}) or {}
        volumes = sample.get("volumes", {}) or {}
        scans = sample.get("scans", {}) or {}

        for segment_id, segment in segments.items():
            total += 1
            if not isinstance(segment, Mapping):
                continue
            segment_date = _creation_date(segment)
            ovid = segment.get("original_volume_id")
            volume = volumes.get(ovid) if ovid else None
            scan_id = volume.get("scan_id") if isinstance(volume, Mapping) else None
            scan = scans.get(scan_id) if scan_id else None
            scan_date = _creation_date(scan)
            if segment_date is None or scan_date is None:
                continue
            comparable += 1
            if scan_date > segment_date:
                own_ids = set(_VOLUME_ID_IN_PATH.findall(json.dumps(segment.get("data"))))
                findings.append(
                    Finding(
                        sample=sample_name,
                        segment_id=str(segment_id),
                        segment_date=segment_date.date().isoformat(),
                        original_volume_id=str(ovid),
                        scan_id=str(scan_id),
                        scan_date=scan_date.date().isoformat(),
                        gap_days=(scan_date - segment_date).days,
                        declared_id_in_own_paths=str(ovid) in own_ids,
                    )
                )
    return findings, total, comparable


def write_csv(findings: Sequence[Finding], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample",
                "segment_id",
                "segment_date",
                "original_volume_id",
                "scan_id",
                "scan_date",
                "gap_days",
                "declared_id_in_own_paths",
            ]
        )
        for f in findings:
            writer.writerow(
                [
                    f.sample,
                    f.segment_id,
                    f.segment_date,
                    f.original_volume_id,
                    f.scan_id,
                    f.scan_date,
                    f.gap_days,
                    f.declared_id_in_own_paths,
                ]
            )


def _report(findings: Sequence[Finding], total: int, comparable: int) -> None:
    by_sample: dict[str, int] = {}
    for f in findings:
        by_sample[f.sample] = by_sample.get(f.sample, 0) + 1
    print(
        f"segments: {total} total, {comparable} with a resolvable scan date; "
        f"{len(findings)} declare a volume scanned AFTER the segment"
    )
    if by_sample:
        grouped = ", ".join(f"{k} x{v}" for k, v in sorted(by_sample.items()))
        print(f"by sample: {grouped}")
        print()
        for f in sorted(findings, key=lambda r: (r.sample, r.segment_id)):
            pointer = "id in own paths" if f.declared_id_in_own_paths else "id NOT in own paths"
            print(
                f"  {f.sample:<16} seg {f.segment_id}  {f.segment_date}  "
                f"-> vol {f.original_volume_id} / scan {f.scan_id} {f.scan_date}  "
                f"(+{f.gap_days}d, {pointer})"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--metadata-url",
        default=DEFAULT_METADATA_URL,
        help="URL of the catalog metadata.json (default: the public open-data copy).",
    )
    source.add_argument(
        "--metadata-file",
        default=None,
        help="Read a local metadata.json (or .json.gz) instead of downloading.",
    )
    parser.add_argument("--csv", default=None, help="Write the findings to this CSV path.")
    parser.add_argument(
        "--quiet", action="store_true", help="Only print the summary line."
    )
    args = parser.parse_args(argv)

    metadata = load_metadata(
        None if args.metadata_file else args.metadata_url, args.metadata_file
    )
    findings, total, comparable = find_future_scan_segments(metadata)

    if args.quiet:
        print(
            f"segments: {total} total, {comparable} checked, {len(findings)} with a "
            f"scan newer than the segment"
        )
    else:
        _report(findings, total, comparable)
    if args.csv:
        write_csv(findings, args.csv)
        print(f"\nwrote {len(findings)} rows to {args.csv}")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
