#!/usr/bin/env python3
"""Smoke the rendering gate with Python site packages disabled."""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_render_valgrind_ci  # noqa: F401
from native_thread_sync_replay import NativeReplayEngine
from thread_sync_trace import parse_drd_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-engine", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "drd.log"
        trace.write_text(
            "New segment for thread 1 with vc [1: 1]\n"
            "SCHED[1]: entering VG_(scheduler)\n"
            "SCHED[1]: exiting VG_(scheduler)\n"
        )
        parsed = parse_drd_trace(trace)
        if not parsed.events or parsed.unresolved_happens_before != 0:
            raise RuntimeError("dependency-free DRD parser smoke failed")

    profiles = {
        1: {
            "Ir": 80,
            "Dr": 20,
            "Dw": 10,
            "D1mr": 2,
            "D1mw": 1,
            "DLmr": 1,
            "DLmw": 0,
            "Bcm": 2,
            "Bim": 1,
        }
    }
    model = {
        "feature_names": [
            "non_data_instructions",
            "data_reads",
            "data_writes",
            "l1_data_misses",
            "last_level_data_misses",
            "branch_misses",
            "branch_weighted_l1_misses",
        ],
        "coefficients_ns": [1.0] * 7,
        "stall_overlap_fraction": 0.0,
    }
    expected = 50.0 + 20.0 + 10.0 + 3.0 + 1.0 + 3.0 + 9.0 / 80.0
    with NativeReplayEngine(args.replay_engine) as engine:
        costs, total = engine.model_profile_costs(profiles, model)
    if not math.isclose(costs[1], expected, rel_tol=1e-14) or not math.isclose(
        total, expected, rel_tol=1e-14
    ):
        raise RuntimeError("native event-cost scoring smoke failed")


if __name__ == "__main__":
    main()
