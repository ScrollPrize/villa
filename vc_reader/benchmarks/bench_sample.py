"""Point reads and sampling: read_points, sample(order=0) and sample(order=1), warm.

Measures the CPU cost of the point paths on local data that is already in the
page cache (one untimed repetition first), through memory maps and through the
memory cache. Point sets, all inside one stored 1024^3 region:

* ``random``: points uniform over the region (512 chunks, no locality).
* ``patch``: a surface-like sampling grid, 16 layers of 512 x 512 points on a
  tilted plane with non-integer spacing (an ink-detection patch).
* ``local``: points uniform in a 96^3 box (a few chunks, cache-resident).

Every scenario's output is hashed. ``--compare`` checks the hashes against an
earlier results file, so a speedup can be shown to return exactly the same
values as the build it is compared with.

    python benchmarks/bench_sample.py
    python benchmarks/bench_sample.py --reps 7 --compare results/sample-YYYYMMDD-HHMMSS.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from _common import digest, environment

from vc_reader.reader import ZarrReader

DEFAULT_ARRAY = ("/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/"
                 "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2")
REGION = ((9000, 3600, 3600), 1024)  # start (z, y, x), edge: stored throughout at level 2


def point_sets(start, edge, scale: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    s = np.asarray(start, np.float64)
    n = int(1_000_000 * scale)
    random = s + rng.uniform(0, edge - 1, size=(n, 3))
    # 16 x 512 x 512 grid on a tilted plane: in-plane axes u, v and a normal w.
    u = np.array([0.1, 0.2, 0.97]); u /= np.linalg.norm(u)
    w = np.array([0.95, 0.3, 0.0]); w -= u * (u @ w); w /= np.linalg.norm(w)
    v = np.cross(w, u)
    side = int(512 * scale ** 0.5)
    a, b, c = np.meshgrid(np.arange(16), np.arange(side), np.arange(side), indexing="ij")
    origin = s + edge / 2 - 0.73 * side / 2 * (u + v)
    patch = origin + 0.73 * (b[..., None] * v + c[..., None] * u) + 0.5 * a[..., None] * w
    local = s + 300 + rng.uniform(0, 95, size=(n, 3))
    return {"random": random, "patch": patch.reshape(-1, 3), "local": local}


OPS = {
    "read_points": lambda r, p: r.read_points(p),
    "nearest": lambda r, p: r.sample(p, order=0),
    "linear": lambda r, p: r.sample(p, order=1),
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--array", default=DEFAULT_ARRAY)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the number of points")
    ap.add_argument("--only", default="", help="comma-separated scenario prefixes, e.g. random-linear")
    ap.add_argument("--compare", help="results JSON whose output hashes must match")
    ap.add_argument("--label", default="", help="free-form note stored with the results")
    args = ap.parse_args(argv)

    sets = point_sets(*REGION, args.scale)
    for name, pts in sets.items():
        sets[name] = np.ascontiguousarray(pts)
    readers = {
        "mmap": ZarrReader.open(args.array),
        "memory": ZarrReader.open(args.array, memory_cache=4 << 30),
    }
    only = [o for o in args.only.split(",") if o]
    rows = []
    for set_name, pts in sets.items():
        ipts = np.rint(pts).astype(np.int64)
        for op_name, op in OPS.items():
            for reader_name, r in readers.items():
                scenario = f"{set_name}-{op_name}-{reader_name}"
                if only and not any(scenario.startswith(o) for o in only):
                    continue
                p = ipts if op_name == "read_points" else pts
                out = op(r, p)  # warm-up: page cache, memory cache, output pool
                h = digest(out)
                del out
                times = []
                for _ in range(args.reps):
                    t = time.perf_counter()
                    out = op(r, p)
                    times.append(time.perf_counter() - t)
                    del out
                med = statistics.median(times)
                rows.append(dict(scenario=scenario, points=len(p), reps=args.reps, seconds_min=min(times),
                                 seconds_median=med, seconds_max=max(times), mpts_s_median=len(p) / med / 1e6,
                                 digest=h))
                print(f"  {scenario:<28} {len(p):>9} pts  median {med * 1e3:8.1f} ms "
                      f"({min(times) * 1e3:.1f}-{max(times) * 1e3:.1f})  {len(p) / med / 1e6:7.1f} Mpts/s",
                      flush=True)

    ok = True
    if args.compare:
        before = {r["scenario"]: r for r in json.loads(Path(args.compare).read_text())["rows"]}
        print(f"\n| scenario | points | before ms | after ms | speedup | same output |\n|---|---|---|---|---|---|")
        for r in rows:
            b = before.get(r["scenario"])
            if b is None:
                continue
            same = b["digest"] == r["digest"]
            ok &= same
            print(f"| {r['scenario']} | {r['points']} | {b['seconds_median'] * 1e3:.1f} | "
                  f"{r['seconds_median'] * 1e3:.1f} | {b['seconds_median'] / r['seconds_median']:.2f}x | "
                  f"{'yes' if same else 'NO'} |")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"sample-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(dict(command=" ".join([sys.executable, *sys.argv]), environment=environment(),
                                    array=args.array, region=REGION, label=args.label, rows=rows), indent=1))
    print(f"\nwrote {path}")
    if not ok:
        raise SystemExit("output differs from the compared results")


if __name__ == "__main__":
    main()
