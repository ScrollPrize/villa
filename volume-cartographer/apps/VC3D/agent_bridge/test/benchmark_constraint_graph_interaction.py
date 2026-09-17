"""Measure UI-thread zoom response on an already loaded Constraint Graph.

Wait for proposal discovery to finish, choose Fit view for a repeatable camera,
then run this against an isolated VC3D agent bridge. This measures request
latency, including synchronous UI work; it does not measure completed CT frames.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

from bridge_client import BridgeClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--viewer", default="v6")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--scale", type=float)
    args = parser.parse_args()
    if args.iterations < 2 or args.iterations % 2:
        parser.error("iterations must be positive and even, to restore the zoom")
    if args.scale is not None and (not math.isfinite(args.scale) or args.scale <= 0):
        parser.error("scale must be finite and positive")
    client = BridgeClient(args.socket)
    state, _ = client.call("viewer.zoom", {"viewer": args.viewer, "factor": 1})
    if args.scale is not None:
        state, _ = client.call("viewer.zoom", {
            "viewer": args.viewer, "factor": args.scale / state["scale"]})
    samples = []
    for i in range(args.iterations):
        factor = 1.05 if i % 2 == 0 else 1 / 1.05
        expected_scale = state["scale"] * factor
        state, elapsed = client.call("viewer.zoom", {
            "viewer": args.viewer, "factor": factor}, timeout=60)
        if not math.isclose(state["scale"], expected_scale, rel_tol=1e-5, abs_tol=1e-7):
            raise RuntimeError("Camera changed during the benchmark; rerun without other interaction")
        samples.append(elapsed * 1000)
        time.sleep(.1)
    report = {
        "iterations": len(samples), "scale": state["scale"],
        "mean_ms": statistics.mean(samples), "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[math.ceil(.95 * len(samples)) - 1],
        "samples_ms": samples,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples_ms"}))


if __name__ == "__main__":
    main()
