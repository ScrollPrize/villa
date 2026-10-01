#!/usr/bin/env python3
"""Freeze the reviewed 32-piece failure union into a locked test split."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vesuvius/src"))

from vesuvius.neural_tracing.fiber_trace_3d.reviewed_failure_test import materialize_failure_union


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("summaries", nargs="+", type=Path)
    args = parser.parse_args()
    manifest = materialize_failure_union(args.summaries, args.destination)
    print(f"Materialized {manifest['expected_count']} reviewed failure pieces at {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
