# /// script
# requires-python = ">=3.11"
# dependencies = ["aiohttp"]
# ///
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
E2E = REPO / "tools" / "e2e.py"
WORKFLOWS = REPO / "tests" / "workflows.e2e.py"

parser = argparse.ArgumentParser(description="Two viewer trees through tests/workflows.e2e.py: timings and pixels.")
parser.add_argument("--a", type=Path, required=True, help="first viewer tree (its serve.py serves it)")
parser.add_argument("--b", type=Path, required=True, help="second viewer tree")
parser.add_argument("--runs", type=int, default=3, help="runs of each tree")
parser.add_argument("--out", type=Path, required=True, help="directory for every run's output and compare.json")
args = parser.parse_args()

runs = {"a": [], "b": []}
for i in range(1, args.runs + 1):
    order = [("a", args.a), ("b", args.b)]
    for label, tree in order if i % 2 == 1 else order[::-1]:
        out = args.out / f"{label}-{i}"
        print(f"{label} run {i}: {tree}", flush=True)
        result = subprocess.run([sys.executable, str(E2E), "--web", str(tree), "--out", str(out), str(WORKFLOWS)], capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"{label} run {i} stopped with {result.returncode}\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
        runs[label].append(json.loads((out / "workflows" / "workflows.json").read_text()))

names = []
for label in runs:
    for run in runs[label]:
        names += [name for name in {**run["timings"], **run["measures"]} if name not in names]
rows = []
print(f"\n{'':52s} {'a median':>9s} {'a range':>15s} {'b median':>9s} {'b range':>15s} {'b - a':>8s}  verdict")
for name in names:
    values = {label: [{**run["timings"], **run["measures"]}[name] for run in runs[label]] for label in runs}
    a, b = values["a"], values["b"]
    diff = statistics.median(b) - statistics.median(a)
    beyond = abs(diff) > max(max(a) - min(a), max(b) - min(b))
    verdict = "b higher" if min(b) > max(a) and beyond else "b lower" if max(b) < min(a) and beyond else "within noise"
    rows.append({"name": name, "a": a, "b": b, "verdict": verdict})
    print(f"{name:52s} {statistics.median(a):9.2f} {min(a):7.2f}-{max(a):<7.2f} {statistics.median(b):9.2f} {min(b):7.2f}-{max(b):<7.2f} {diff:+8.2f}  {verdict}")

pixels = []
keys = []
for label in runs:
    for run in runs[label]:
        keys += [key for key in run["pixels"] if key not in keys]
print(f"\n{'step/pane':28s} {'a runs agree':>13s} {'b runs agree':>13s} {'a = b':>6s}")
for key in keys:
    hashes = {label: [run["pixels"].get(key) for run in runs[label]] for label in runs}
    same = {label: len(set(hashes[label])) == 1 and None not in hashes[label] for label in runs}
    across = None not in hashes["a"] + hashes["b"] and set(hashes["a"]) == set(hashes["b"])
    pixels.append({"pane": key, **hashes, "a runs agree": same["a"], "b runs agree": same["b"], "a = b": across})
    print(f"{key:28s} {str(same['a']):>13s} {str(same['b']):>13s} {str(across):>6s}"
          + ("" if same["a"] and same["b"] else f"   distinct a {len(set(hashes['a']))}, b {len(set(hashes['b']))}"))

(args.out / "compare.json").write_text(json.dumps({"a": str(args.a), "b": str(args.b), "runs": args.runs, "rows": rows, "pixels": pixels}, indent=1))
print(f"\nwrote {args.out / 'compare.json'}")
