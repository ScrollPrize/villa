#!/usr/bin/env python3
"""Aggregate clang -ftime-trace output across a whole build into one report.

Build with the dev-profile-clang preset (VC_PROFILE_COMPILE=ON), then point this
at the build dir:

    cmake --preset dev-profile-clang
    cmake --build --preset dev-profile-clang
    scripts/analyze_build_profile.py build/dev-profile-clang

Each TU emits a <tu>.json trace next to its .o. Clang writes these in Chrome
Trace format. IMPORTANT: most events are *complete* events ("ph":"X" with a
"dur"), but Source (header) events are *async* ("ph":"b"/"e" paired by "id") and
carry NO "dur" — you must pair begin/end to get their span. Summing a missing
"dur" silently yields 0, which is the classic mis-parse this script guards
against (see _self_check).

Reports, aggregated across every TU:
  - slowest translation units (ExecuteCompiler)
  - frontend vs backend split
  - heaviest headers by inclusive parse time (top-level includes only, so a
    header included by another isn't double-counted within a TU)
  - heaviest template instantiations
  - per-category totals (Frontend, Backend, Optimizer, CodeGen, ...)
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict


def load_events(path):
    """Return the traceEvents list, or None if the file isn't a usable trace."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    ev = data.get("traceEvents")
    return ev if isinstance(ev, list) and ev else None


def complete_events(events):
    """Yield (name, dur_us, detail, ts, end_ts) for *complete* ('X') events."""
    for e in events:
        if e.get("ph") != "X":
            continue
        dur = e.get("dur", 0)
        det = (e.get("args") or {}).get("detail", "")
        ts = e.get("ts", 0)
        yield e.get("name", ""), dur, det, ts, ts + dur


def source_spans(events):
    """Pair async Source begin/end ('b'/'e') events by id -> list of spans.

    Returns [(detail, ts, end_ts, dur_us)] for each header parse. Async events
    have no 'dur'; the span is end.ts - begin.ts.
    """
    begins = {}  # id -> (detail, ts)
    spans = []
    for e in events:
        if e.get("cat") != "Source" and e.get("name") != "Source":
            continue
        ph = e.get("ph")
        eid = e.get("id")
        if ph == "b":
            begins[eid] = ((e.get("args") or {}).get("detail", ""), e.get("ts", 0))
        elif ph == "e" and eid in begins:
            det, ts = begins.pop(eid)
            spans.append((det, ts, e.get("ts", 0), e.get("ts", 0) - ts))
    return spans


def top_level_spans(spans):
    """Keep only spans not nested inside another span (within one TU).

    A header's inclusive cost is its full span; counting nested includes too
    would multiply-count. Sort by start asc / end desc, then drop any span
    fully contained in an already-kept one.
    """
    spans = sorted(spans, key=lambda s: (s[1], -s[2]))
    kept = []
    kept_intervals = []
    for det, ts, te, dur in spans:
        if any(ots <= ts and te <= ote for ots, ote in kept_intervals):
            continue
        kept.append((det, dur))
        kept_intervals.append((ts, te))
    return kept


def shorten(path, build_dir):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(build_dir)))
    return (
        path.replace(repo + "/", "")
        .replace(f"{build_dir}/_deps/", "_deps/")
        .replace("/usr/lib/gcc/x86_64-linux-gnu/15/../../../../include/", "<sys>/")
        .replace("/usr/include/", "<sys>/")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    traces = [
        f
        for f in glob.glob(os.path.join(args.build_dir, "**", "*.json"), recursive=True)
        if "CMakeFiles" in f and (f.endswith(".cpp.json") or f.endswith(".c.json"))
    ]
    if not traces:
        sys.exit(f"no *.cpp.json traces under {args.build_dir} — build with "
                 "the dev-profile-clang preset (VC_PROFILE_COMPILE=ON)")

    tu_total = {}              # tu -> ExecuteCompiler seconds
    cat_total = defaultdict(float)
    header_total = defaultdict(float)
    header_tus = defaultdict(int)
    inst_total = defaultdict(float)

    parsed = 0
    for f in traces:
        ev = load_events(f)
        if ev is None:
            continue
        parsed += 1
        tu = os.path.basename(f)[:-5]  # strip .json
        for name, dur, det, _ts, _te in complete_events(ev):
            s = dur / 1e6
            if name == "ExecuteCompiler":
                tu_total[tu] = s
            elif name.startswith("Total "):
                cat_total[name[len("Total "):]] += s
            elif name in ("InstantiateClass", "InstantiateFunction") and det:
                inst_total[det] += s
        for det, dur in top_level_spans(source_spans(ev)):
            if not det:
                continue
            header_total[det] += dur / 1e6
            header_tus[det] += 1

    # ---- self-check: refuse to emit a silently-empty report ----
    grand = sum(tu_total.values())
    _self_check(parsed, grand, header_total, tu_total)

    print(f"# Build profile — {parsed} TUs, {grand:.0f}s total CPU compile\n")

    print("## Slowest translation units (s)")
    for tu, s in sorted(tu_total.items(), key=lambda x: -x[1])[:args.top]:
        print(f"  {s:6.1f}  {tu}")

    print("\n## Phase totals (s, summed across TUs)")
    for cat in ("Frontend", "Backend", "Optimizer", "CodeGen Passes",
                "Source", "Instantiate Class", "Instantiate Function"):
        if cat in cat_total:
            print(f"  {cat_total[cat]:7.0f}  {cat}")

    # All headers incl. third-party — a heavy 3rd-party header is a candidate
    # to replace or drop (e.g. doctest was swapped for an in-tree framework).
    print("\n## Heaviest headers — all, incl. third-party (s, #TUs)")
    for det, s in sorted(header_total.items(), key=lambda x: -x[1])[:args.top]:
        print(f"  {s:6.0f}  x{header_tus[det]:<4} {shorten(det, args.build_dir)}")

    # First-party = under the repo but not the build dir (which holds _deps).
    # These are the headers we can actually edit; ranking by cost x #TUs finds
    # the highest-leverage include-trimming targets.
    repo = os.path.dirname(os.path.dirname(os.path.abspath(args.build_dir)))
    bdir = os.path.abspath(args.build_dir)
    ours = {h: s for h, s in header_total.items()
            if os.path.abspath(h).startswith(repo) and not os.path.abspath(h).startswith(bdir)}
    print("\n## Heaviest first-party headers (ours to fix; s, #TUs)")
    for det, s in sorted(ours.items(), key=lambda x: -x[1])[:args.top]:
        print(f"  {s:6.0f}  x{header_tus[det]:<4} {shorten(det, args.build_dir)}")

    print("\n## Heaviest template instantiations (s, summed)")
    for det, s in sorted(inst_total.items(), key=lambda x: -x[1])[:args.top]:
        print(f"  {s:6.1f}  {det[:100]}")


def _self_check(parsed, grand, header_total, tu_total):
    problems = []
    if parsed == 0:
        problems.append("parsed 0 trace files")
    if grand <= 0:
        problems.append("total ExecuteCompiler time is 0 — 'X' event 'dur' not read")
    if header_total and max(header_total.values()) <= 0:
        problems.append("all header times are 0 — async 'b'/'e' Source pairing failed")
    if tu_total and max(tu_total.values()) <= 0:
        problems.append("all TU times are 0")
    if problems:
        sys.exit("PROFILE PARSE FAILED:\n  - " + "\n  - ".join(problems))


if __name__ == "__main__":
    main()
