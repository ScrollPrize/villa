#!/usr/bin/env python3
"""Report coarse pyramid chunks that are missing although finer levels have data.

Listing-only (no chunk downloads). Exit code is 1 if any hole is found, so it
can gate uploads or run in CI.

Example::

    python -m vesuvius.scripts.check_pyramid_occupancy --anon \\
        s3://vesuvius-challenge-open-data/PHerc0343P/volumes/20250521134555-8.640um-1.2m-116keV-masked.zarr
"""

from __future__ import annotations

import argparse
import json
import sys

from vesuvius.data.pyramid_occupancy import check_pyramid


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stores", nargs="+", help="OME-Zarr store URLs (local path or s3://...)")
    p.add_argument("--anon", action="store_true", help="Unsigned S3 requests (public buckets).")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = p.parse_args(argv)

    bad = False
    report = []
    for store in args.stores:
        for r in check_pyramid(store, anon=args.anon):
            bad |= r.missing > 0
            report.append({"store": store, **r.__dict__, "examples": [list(e) for e in r.examples]})
            if not args.json:
                if r.skipped:
                    status = f"SKIPPED ({r.skipped})"
                elif r.missing:
                    status = f"HOLES {r.missing}/{r.expected} e.g. {[list(e) for e in r.examples[:5]]}"
                else:
                    status = f"ok ({r.expected} expected chunks present)"
                print(f"{store} level {r.fine} -> {r.coarse}: {status}")
    if args.json:
        print(json.dumps(report, indent=2))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
