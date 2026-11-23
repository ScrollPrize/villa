#!/usr/bin/env python3
"""Lays out benchmark results side by side with the manifest's saved results.

    compare.py [--config benchmarks.json] [RESULTS.jsonl] [--metric NAME]...

Rows are (benchmark, case, cache state, step, metric). Columns are the means of
each pipeline in RESULTS (this run's results file, optional), then of each entry
of the manifest's "results" list:

    "baseline": "main-1101",
    "results": [
        {"label": "main-1101", "path": "baselines/main-1101.jsonl"},
        {"label": "cpu-before-prefetch-fix", "path": "/somewhere/results.jsonl"}
    ]

Paths are relative to the manifest. Every column but the baseline's is followed
by its change against the baseline in percent (negative: lower than the
baseline). The baseline is the entry "baseline" names, else the first; a column
is set against the baseline's same pipeline, or the baseline's only pipeline when
it has just one (e.g. a saved origin/main run, "chunkcache"). The latest line of
each (row, pipeline) in a file wins, so a file that runs append to works as is.
Configuration differences against the baseline (parameters, data, cache state)
are listed under the table: they make a comparison one of different workloads.
"""

import argparse
import collections
import json
import os
import sys

DEFAULT_METRICS = ("settled_ms", "first_frame_ms", "wall_ms", "ms", "presented_fps")
# Configuration that describes a run's outcome or machine-local paths, not its workload.
VOLATILE = {
    "volume_open_ms", "decoded_cache_mb", "decoded_chunks", "known_empty_chunks", "fetched_chunks",
    "fetched_mb", "config", "cache", "volume_cache_root", "partial_seeded_files", "tool", "backend", "work_dir", "offline_misses",
}
# Parameters only some pipelines read: not a difference when the other side lacks them.
PIPELINE_ONLY = {"gpu_cache_gb", "cuda", "cuda-cache-gb", "remote_offline"}


class Results:
    def __init__(self, label, path, metrics):
        self.label = label
        self.rows = collections.OrderedDict()  # (bench, case, state, step, metric) -> {pipeline: mean}
        self.configs = {}                      # (bench, case, state, pipeline) -> config
        try:
            f = open(path)
        except OSError as e:
            sys.exit(f"results {label or path}: {e}")
        with f:
            for number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    sys.exit(f"{path}:{number}: {e}")
                run = (r["benchmark"], r.get("case", ""), r.get("cache_state", "cached"))
                self.configs[run + (r["pipeline"],)] = r.get("config", {})
                if r["metric"] in metrics:
                    self.rows.setdefault(run + (r["step"], r["metric"]), {})[r["pipeline"]] = r["mean"]
        self.pipelines = list(dict.fromkeys(p for values in self.rows.values() for p in values))

    def counterpart(self, pipeline):
        """The pipeline of these results a column of `pipeline` is set against."""
        if pipeline in self.pipelines:
            return pipeline
        return self.pipelines[0] if len(self.pipelines) == 1 else None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="?", help="this run's results file")
    parser.add_argument("--config", default=os.path.join(here, "benchmarks.json"))
    parser.add_argument("--metric", action="append", help=f"metrics to show (default: {', '.join(DEFAULT_METRICS)})")
    args = parser.parse_args()
    metrics = set(args.metric or DEFAULT_METRICS)

    with open(args.config) as f:
        manifest = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(args.config))
    sources = []
    if args.results and os.path.exists(args.results):
        sources.append(Results("", args.results, metrics))
    elif args.results:
        print(f"note: {args.results} does not exist yet; showing saved results only", file=sys.stderr)
    saved = manifest.get("results", [])
    labels = set()
    for entry in saved:
        label, path = entry.get("label"), entry.get("path")
        if not label or not path:
            sys.exit(f"{args.config}: every \"results\" entry needs a \"label\" and a \"path\"")
        if label in labels:
            sys.exit(f"{args.config}: results label {label!r} is used twice")
        labels.add(label)
        sources.append(Results(label, os.path.join(base_dir, path), metrics))
    baseline = None
    if saved:
        name = manifest.get("baseline") or saved[0]["label"]
        baseline = next((s for s in sources if s.label == name), None)
        if baseline is None:
            sys.exit(f"{args.config}: \"baseline\" {name!r} is not a label of \"results\"")
    if not sources:
        sys.exit("nothing to compare: no results file and no saved \"results\" in the manifest")

    keys = list(dict.fromkeys(k for s in sources for k in s.rows))
    columns = []  # (header, function of row key -> value, is a delta)
    for s in sources:
        for p in s.pipelines:
            header = f"{s.label}:{p}" if s.label else p
            columns.append((header, lambda k, s=s, p=p: s.rows.get(k, {}).get(p), False))
            if baseline is not None and s is not baseline and baseline.counterpart(p):
                b = baseline.counterpart(p)
                columns.append((f"d% {header}",
                                lambda k, s=s, p=p, b=b: delta(s.rows.get(k, {}).get(p),
                                                               baseline.rows.get(k, {}).get(b)), True))

    if baseline is not None:
        print(f"baseline: {baseline.label} ({', '.join(baseline.pipelines)})")
    width = max([16] + [len(h) + 2 for h, _, _ in columns])
    print(f'{"benchmark":18} {"case":20} {"state":8} {"step":40} {"metric":15}' +
          "".join(f"{h:>{width}}" for h, _, _ in columns))
    for k in keys:
        cells = []
        for _, value, is_delta in columns:
            v = value(k)
            if v is None:
                cells.append(f'{"-":>{width}}')
            else:
                cells.append(f"{v:>+{width}.1f}" if is_delta else f"{v:>{width}.2f}")
        print(f"{k[0]:18} {k[1]:20} {k[2]:8} {k[3]:40} {k[4]:15}" + "".join(cells))

    if baseline is None:
        return
    for s in sources:
        if s is baseline:
            continue
        for (bench, case, state, p), config in s.configs.items():
            b = baseline.counterpart(p)
            base = baseline.configs.get((bench, case, state, b)) if b else None
            if base is None:
                continue
            shared = (set(config) & set(base)) - VOLATILE
            diff = sorted(key for key in shared if config[key] != base[key])
            one_side = sorted((set(config) ^ set(base)) - VOLATILE - PIPELINE_ONLY)
            if diff or one_side:
                print(f"\nnote: {s.label or 'this run'} {p} differs from baseline {baseline.label} {b} "
                      f"in {bench}/{case}/{state}:")
                for key in diff:
                    print(f"  {key}: {base[key]} -> {config[key]}")
                if one_side:
                    print(f"  recorded on one side only: {', '.join(one_side)}")


def delta(current, base):
    if current is None or base is None or base == 0:
        return None
    return 100.0 * (current - base) / base


if __name__ == "__main__":
    main()
