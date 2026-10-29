#!/usr/bin/env python3
"""Summarize a VC3D publication-metrics log (``VC3D --publication-metrics <file>``).

The log holds nested spans from the line-annotation publication path (clicks,
landings, saves, commits with their hold/act phases, the Committed handler and
its consumers), the hub worker's sync phases and lease waits, and a GUI-thread
heartbeat whose every sample is recorded. See apps/VC3D/PublicationMetrics.hpp
for the line format.

Reported:
  * per span name: count, mean/p50/p95/max of inclusive and exclusive time
    (exclusive = inclusive minus the children on the same thread);
  * heartbeat gap distribution over every sample, and each gap above the
    thresholds attributed to the GUI spans that overlapped it plus the deferred
    consumers it caused (``cause=<span id>`` attributes, followed transitively);
  * commit/act counts, journal correlation between ``commit.act`` and the
    Committed events that delivered its entries (delivery latency, events that
    aggregate several commits), hold waits, and commits whose hold overlapped a
    worker sync;
  * the decision rule from the plan: p95 of complete click and landing spans
    under 100 ms and no heartbeat gap over 250 ms attributed to click, landing,
    commit, the Committed handler or the consumers they caused.

The measurement window is [first click or landing, quit marker): the quit
marker (``header ... quit=1``, written when the stop was requested or at
shutdown) ends it, and everything after it is shutdown work, excluded from
timing, attribution, coverage and completeness alike. Work that started inside
the window must have finished inside it.

Percentiles interpolate linearly between order statistics (numpy's default);
thresholds are strict (``< 100 ms``). Durations are inclusive.

Completeness (every item must hold, or the verdict is INCOMPLETE):
  * the run has a quit marker and ended with its end marker, with ``failures=0``
    and no dropped records; no malformed or duplicate span records;
  * no publication failed inside the window: no ``save failed=1``, no landing
    ``outcome=failed``, no ``commit.act`` without both journal receipts;
  * every relevant span that began inside the window ended inside it, and the
    heartbeat covers the last one;
  * every commit's journal entries were delivered by a Committed event of the
    same attachment, inside the window;
  * every catalog request announced inside the window was served by a
    ``consumer.catalog`` span (same or later request) inside the window;
  * every click that placed a control (a ``click.place`` child with a solve
    epoch) has a later final applied landing on its own pane (the click's
    strip surface minus ``_line_surface``) with a solve epoch at least that;
  * with ``--timeline``: the same for every placement the fixture made.

Exit status: 0 PASS, 1 FAIL (complete, but the rule or a requirement failed),
2 INCOMPLETE (the log cannot be trusted; ``--allow-incomplete`` only keeps the
report going for diagnosis, the verdict is never PASS).

Usage: publication_metrics.py <log> [--json <out.json>] [--click-p95-ms 100]
       [--gap-ms 250] [--min-clicks N] [--min-landings N] [--min-overlaps N]
       [--timeline timeline.json] [--require-span NAME ...]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Span:
    id: int
    parent: int
    thread: str
    name: str
    begin: int
    end: int | None = None
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)
    unbalanced: bool = False
    parent_span: "Span | None" = None

    @property
    def complete(self) -> bool:
        return self.end is not None and not self.unbalanced

    @property
    def inclusive_ns(self) -> int:
        return (self.end or self.begin) - self.begin

    @property
    def exclusive_ns(self) -> int:
        # Children on another thread (a worker span nested under a GUI id)
        # do not consume this span's time.
        return self.inclusive_ns - sum(c.inclusive_ns for c in self.children if c.thread == self.thread)

    def ancestors(self):
        s = self
        while s is not None:
            yield s
            s = s.parent_span

    def int_attr(self, key: str) -> int | None:
        try:
            return int(self.attrs[key])
        except (KeyError, ValueError):
            return None


def parse_attrs(words: list[str]) -> dict[str, str]:
    out = {}
    for w in words:
        k, _, v = w.partition("=")
        out[k] = v
    return out


@dataclass
class Log:
    spans: dict[int, Span]
    heartbeats: list[tuple[int, int]]
    events: list[dict]
    headers: list[dict]
    malformed: int = 0
    duplicate_ids: int = 0


def parse(path: str) -> Log:
    log = Log({}, [], [], [])
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            parts = raw.split()
            if not parts:
                continue
            kind = parts[0]
            try:
                if kind == "hb":
                    log.heartbeats.append((int(parts[1]), int(parts[2])))
                elif kind == "span":
                    side, sid, parent, thread, ns, name = parts[1], int(parts[2]), int(parts[3]), parts[4], int(parts[5]), parts[6]
                    attrs = parse_attrs(parts[7:])
                    if side == "B":
                        if sid in log.spans:
                            log.duplicate_ids += 1
                            continue
                        log.spans[sid] = Span(sid, parent, thread, name, ns, attrs=attrs)
                    elif side == "E":
                        s = log.spans.get(sid)
                        # An end must match its begin (thread, nesting) and not precede it.
                        if s is None or s.end is not None or s.thread != thread or s.parent != parent or ns < s.begin:
                            log.malformed += 1
                            continue
                        s.end = ns
                        s.attrs.update(attrs)
                        if attrs.get("unbalanced") == "1":
                            s.unbalanced = True
                    else:
                        log.malformed += 1
                elif kind == "event":
                    log.events.append({"parent": int(parts[1]), "thread": parts[2], "ns": int(parts[3]),
                                       "name": parts[4], **parse_attrs(parts[5:])})
                elif kind == "header":
                    timed = len(parts) > 1 and parts[1].isdigit()
                    log.headers.append({"ns": int(parts[1]) if timed else 0, **parse_attrs(parts[2:] if timed else parts[1:])})
                else:
                    log.malformed += 1
            except (IndexError, ValueError):
                log.malformed += 1
    for s in log.spans.values():
        p = log.spans.get(s.parent)
        if p is s:
            log.malformed += 1
            p = None
        s.parent_span = p
    # The parent graph must be a forest: a cycle is a malformed record, and
    # the link that closes it is dropped so that walks terminate. Child lists
    # are built from the repaired links only.
    for s in log.spans.values():
        seen = {s.id}
        a = s.parent_span
        while a is not None:
            if a.id in seen:
                log.malformed += 1
                s.parent_span = None
                break
            seen.add(a.id)
            a = a.parent_span
    for s in log.spans.values():
        if s.parent_span is not None:
            s.parent_span.children.append(s)
    return log


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def stats_ms(values_ns: list[int]) -> dict:
    ms = [v / 1e6 for v in values_ns]
    if not ms:
        return {"n": 0}
    return {"n": len(ms), "mean": statistics.fmean(ms), "p50": pct(ms, .5), "p95": pct(ms, .95), "max": max(ms)}


def fmt(st: dict) -> str:
    if st.get("n", 0) == 0:
        return "n=0"
    return f"n={st['n']:<5d} mean={st['mean']:8.2f} p50={st['p50']:8.2f} p95={st['p95']:8.2f} max={st['max']:9.2f}"


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


GATE_ROOTS = ("click", "landing", "commit", "committed", "save", "display_save", "branch_saves")
# A click carries the strip surface it was made on; the landing that
# publishes it carries the pane (the strip's prefix).
STRIP_SUFFIX = "_line_surface"


def click_pane(click: Span) -> str:
    surface = click.attrs.get("surface", "")
    return surface[:-len(STRIP_SUFFIX)] if surface.endswith(STRIP_SUFFIX) else surface


def place_epoch(click: Span) -> int | None:
    epochs = [c.int_attr("epoch") for c in descendants(click) if c.name == "click.place" and c.int_attr("epoch") is not None]
    return max(epochs) if epochs else None


def same_thread_ancestors(span: Span):
    """The span and its ancestors on the same thread (a worker span nested
    under a GUI id is the worker's work, not the GUI's)."""
    return (a for a in span.ancestors() if a.thread == span.thread)


def gate_relevant(span: Span) -> bool:
    """A span counts toward the decision rule when it, or an ancestor on its
    thread, is a click, landing, commit or Committed-handler span."""
    return any(a.name in GATE_ROOTS for a in same_thread_ancestors(span))


def causes_of(span: Span) -> list[int]:
    raw = span.attrs.get("cause", "")
    out = []
    for part in raw.split(","):
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def descendants(span: Span):
    for c in span.children:
        yield c
        yield from descendants(c)


class Relevance:
    """Transitive attribution: a deferred consumer is gate work when the span
    that scheduled it (its cause), that span's ancestors or descendants, or
    the cause of that span in turn, is. Unknown causes (0, or an id the log
    does not hold) count as caused by the measurement: conservative, it can
    only turn a pass into a failure. Cycles terminate."""

    def __init__(self, spans: dict[int, Span]):
        self.spans = spans
        self.memo: dict[int, bool] = {}

    def span_relevant(self, span: Span, visiting: set[int]) -> bool:
        if span.id in self.memo:
            return self.memo[span.id]
        if span.id in visiting:
            return False
        visiting.add(span.id)
        result = gate_relevant(span) or any(d.name in GATE_ROOTS and d.thread == span.thread for d in descendants(span))
        if not result:
            result = self.causes_relevant(span, visiting)
        visiting.discard(span.id)
        self.memo[span.id] = result
        return result

    def causes_relevant(self, span: Span, visiting: set[int]) -> bool:
        causes = [c for a in same_thread_ancestors(span) for c in causes_of(a)]
        if not causes:
            return any(a.name.startswith("consumer.") for a in same_thread_ancestors(span))
        for c in causes:
            if c == 0 or c not in self.spans:
                return True
            if self.span_relevant(self.spans[c], visiting):
                return True
        return False

    def relevant(self, span: Span) -> bool:
        if gate_relevant(span):
            return True
        if not any(a.name.startswith("consumer.") for a in same_thread_ancestors(span)):
            return False
        return self.causes_relevant(span, set())

    def caused_by_measured(self, span: Span, measured) -> bool:
        """Whether the span's cause chain leads to work that began inside the
        measurement (unknown causes do not count here)."""
        todo = [c for a in same_thread_ancestors(span) for c in causes_of(a)]
        seen: set[int] = set()
        while todo:
            c = todo.pop()
            if c in seen or c not in self.spans:
                continue
            seen.add(c)
            cause = self.spans[c]
            if measured(cause):
                return True
            todo.extend(x for a in same_thread_ancestors(cause) for x in causes_of(a))
        return False


def finite(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"{text!r} is not a finite number")
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--json", help="write the summary as JSON here")
    ap.add_argument("--click-p95-ms", type=finite, default=100.0)
    ap.add_argument("--gap-ms", type=finite, default=250.0, help="forbidden attributed heartbeat gap")
    ap.add_argument("--report-gap-ms", type=finite, default=100.0, help="list every attributed gap above this")
    ap.add_argument("--min-clicks", type=int, default=1, help="complete clicks that placed a control (click.place with an epoch)")
    ap.add_argument("--min-landings", type=int, default=1, help="final applied landings")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="keep reporting an incomplete log (the verdict stays INCOMPLETE, never PASS)")
    ap.add_argument("--min-overlaps", type=int, default=0, help="required commits whose hold overlapped a sync")
    ap.add_argument("--timeline", help="the fixture's timeline.json: every placement must have its final landing")
    ap.add_argument("--require-span", action="append", default=[], metavar="NAME",
                    help="a span name that must occur inside the window (repeatable)")
    args = ap.parse_args()

    log = parse(args.log)
    spans, heartbeats, events, headers = log.spans, log.heartbeats, log.events, log.headers
    summary: dict = {"log": args.log, "headers": headers, "spans": {}, "gaps": {}, "correlation": {}, "gate": {}}
    relevance = Relevance(spans)

    for h in headers:
        print("header:", " ".join(f"{k}={v}" for k, v in h.items() if k != "ns"))

    # ---- the measurement window ------------------------------------------
    quit_marker = max((h["ns"] for h in headers if h.get("quit") == "1" and h["ns"]), default=None)
    window_start = min((s.begin for s in spans.values() if s.name in ("click", "landing")), default=0)
    # Work that began before the quit marker belongs to the measurement.
    measured = lambda s: quit_marker is not None and s.begin < quit_marker  # noqa: E731
    inside = lambda s: measured(s) and s.complete and s.end <= quit_marker  # noqa: E731

    # ---- per-name timing -------------------------------------------------
    by_name: dict[str, list[Span]] = defaultdict(list)
    for s in spans.values():
        by_name[s.name].append(s)
    print("\nspans (ms; exclusive = minus same-thread children; the whole log, shutdown included):")
    print(f"  {'name':28s} {'inclusive':>58s} | {'exclusive':>58s}")
    for name in sorted(by_name):
        complete = [s for s in by_name[name] if s.complete]
        incomplete = len(by_name[name]) - len(complete)
        inc = stats_ms([s.inclusive_ns for s in complete])
        exc = stats_ms([s.exclusive_ns for s in complete])
        summary["spans"][name] = {"inclusive": inc, "exclusive": exc, "incomplete": incomplete}
        tail = f"  (+{incomplete} incomplete)" if incomplete else ""
        print(f"  {name:28s} {fmt(inc):>58s} | {fmt(exc):>58s}{tail}")

    # ---- heartbeat gaps --------------------------------------------------
    gui_spans = sorted((s for s in spans.values() if s.thread == "gui" and s.end is not None), key=lambda s: s.begin)
    gaps_ms = [g / 1e6 for _, g in heartbeats]
    gap_stats = {"samples": len(gaps_ms)}
    if gaps_ms:
        gap_stats.update({"mean": statistics.fmean(gaps_ms), "p50": pct(gaps_ms, .5), "p95": pct(gaps_ms, .95),
                          "p99": pct(gaps_ms, .99), "max": max(gaps_ms),
                          "over_100": sum(g > 100 for g in gaps_ms), "over_250": sum(g > 250 for g in gaps_ms)})
    summary["gaps"]["distribution"] = gap_stats
    print("\nheartbeat gaps (ms, every sample):", " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                                           for k, v in gap_stats.items()))

    def attribute(t0: int, t1: int) -> tuple[list[Span], bool]:
        """GUI spans overlapping [t0, t1] (outermost first); whether the gap
        counts toward the gate (a relevant span, or a consumer one caused).
        Only measured work can be blamed: spans that began inside the window,
        and deferred consumers that measured work scheduled; startup work
        before the first placement and shutdown work after the quit marker
        can overlap a gap that crosses the window's edge without owning it."""
        hit = [s for s in gui_spans if overlaps(s.begin, s.end, t0, t1) and
               ((measured(s) and s.begin >= window_start) or relevance.caused_by_measured(s, measured))]
        relevant = any(relevance.relevant(s) for s in hit)
        return hit, relevant

    # Every gap above the gate threshold is attributed, whatever the
    # reporting threshold; the report lists those above --report-gap-ms.
    # Gated: every gap that overlaps the window, the ones crossing its start
    # or its end included (a stall is not forgiven for straddling the quit
    # marker); gaps entirely before the first placement or entirely after the
    # quit marker are reported, not gated.
    attributed = []
    gate_threshold = min(args.report_gap_ms, args.gap_ms)
    for ts, gap in heartbeats:
        if gap / 1e6 <= gate_threshold:
            continue
        if ts <= window_start or quit_marker is None or ts - gap >= quit_marker:
            where = "(before the first placement)" if ts <= window_start else "(after the quit marker)"
            attributed.append({"ns": ts, "gap_ms": gap / 1e6, "relevant": False, "gated": False, "roots": [where], "spans": []})
            continue
        hit, relevant = attribute(ts - gap, ts)
        roots = sorted({a.name for s in hit for a in s.ancestors() if a.parent_span is None})
        attributed.append({"ns": ts, "gap_ms": gap / 1e6, "relevant": relevant, "gated": True, "roots": roots,
                           "spans": [f"{s.name}#{s.id}" for s in hit[:8]]})
    summary["gaps"]["attributed"] = attributed
    reported = [g for g in attributed if g["gap_ms"] > args.report_gap_ms]
    print(f"\nheartbeat gaps > {args.report_gap_ms:.0f} ms: {len(reported)}")
    for g in reported[:60]:
        flag = "GATE" if g["relevant"] else "    "
        print(f"  {flag} {g['gap_ms']:8.1f} ms  roots={','.join(g['roots']) or '-'}  spans={' '.join(g['spans']) or '-'}")
    if len(reported) > 60:
        print(f"  ... {len(reported) - 60} more")

    # ---- commits, holds, syncs, correlation (inside the window) ----------
    acts = [s for s in by_name.get("commit.act", []) if inside(s)]
    # Publication that did not succeed inside the window: publication roots
    # reporting a failure (save, branch_saves, display_save), failed landings,
    # and act phases that ended without both journal receipts in order and
    # without their attachment (an exception unwound them).
    failed_saves = [s for s in spans.values() if measured(s) and s.name in GATE_ROOTS and s.attrs.get("failed") == "1"]
    failed_landings = [s for s in by_name.get("landing", []) if measured(s) and s.attrs.get("outcome") == "failed"]

    def receipt_ok(a: Span) -> bool:
        b, e = a.int_attr("seq_before"), a.int_attr("seq_after")
        return a.complete and b is not None and e is not None and e >= b and bool(a.attrs.get("attachment"))
    acts_without_receipt = [s for s in by_name.get("commit.act", []) if measured(s) and not receipt_ok(s)]
    # Coverage counts work inside the window: from the first placement on.
    in_window = lambda s: inside(s) and s.begin >= window_start  # noqa: E731
    # Lease holds are the hub worker's "hold" spans on the GUI thread (the
    # session's checked edit takes them; the controller has no phase of its own).
    holds = [s for s in by_name.get("hold", []) if in_window(s)]
    hold_waits = [s for s in by_name.get("hold.wait", []) if in_window(s)]
    # A sync still running at the quit marker owned hub_ up to it.
    syncs = [s for s in by_name.get("sync", []) if measured(s)]
    committed = [s for s in by_name.get("committed", []) if inside(s)]

    # A sync owns hub_ from the end of its own wait: a GUI hold.wait that
    # overlaps that interval waited behind the SFTP work (the contention the
    # fast path removes); a sync.wait overlapping a GUI hold is the other
    # direction and does not count.
    def owned(sync: Span) -> tuple[int, int] | None:
        waits = [c for c in sync.children if c.name == "sync.wait"]
        if waits and waits[0].end is None:
            return None  # still waiting for hub_: it owns nothing yet
        end = sync.end if sync.end is not None else quit_marker
        return (waits[0].end if waits else sync.begin), min(end, quit_marker)
    sync_owned = [o for o in (owned(y) for y in syncs) if o is not None]
    # Coverage: holds attempted while a sync owned hub_ (whether or not they
    # waited: a fast path that did not wait is the behaviour under test).
    holds_during_sync = sum(any(overlaps(h.begin, h.end, a, b) for a, b in sync_owned) for h in holds)
    gui_waits_behind_sync = sum(any(overlaps(w.begin, w.end, a, b) for a, b in sync_owned) for w in hold_waits
                                if w.thread == "gui")
    sync_waits = [s for s in by_name.get("sync.wait", []) if inside(s) and s.inclusive_ns > 1_000_000]
    worker_waits_behind_gui = sum(any(overlaps(w.begin, w.end, h.begin, h.end) for h in holds) for w in sync_waits)
    corr = {"acts": len(acts), "holds": len(holds), "hold_waits": len(hold_waits),
            "hold_wait_ms": stats_ms([s.inclusive_ns for s in hold_waits]),
            "syncs": len(syncs), "holds_during_sync": holds_during_sync, "gui_waits_behind_sync": gui_waits_behind_sync,
            "worker_waits_behind_gui": worker_waits_behind_gui, "committed_events": len(committed),
            "sync_overlap_observed": holds_during_sync > 0}

    # Each act produced journal entries in (seq_before, seq_after]; the Committed
    # event whose [seq_first, seq_last] contains them delivered it.
    latencies = []
    undelivered = 0
    negative = 0
    per_event: dict[int, int] = defaultdict(int)
    for a in acts:
        if not receipt_ok(a):
            continue
        b, e = a.int_attr("seq_before"), a.int_attr("seq_after")
        if e == b:
            continue
        # Sequence numbers are per store: only events of the same attachment
        # match, and the event must contain the whole interval.
        att = a.attrs.get("attachment")
        hit = [c for c in committed if c.attrs.get("attachment") == att and
               c.int_attr("seq_first") is not None and c.int_attr("seq_last") is not None and
               c.int_attr("seq_first") <= b + 1 and c.int_attr("seq_last") >= e and c.begin >= a.end]
        if not hit:
            undelivered += 1
            continue
        c = min(hit, key=lambda c: c.begin)
        per_event[c.id] += 1
        latency = c.begin - a.end
        if latency < 0:
            negative += 1
        latencies.append(latency)
    aggregated = sum(1 for n in per_event.values() if n > 1)
    corr.update({"delivery_latency_ms": stats_ms(latencies), "events_aggregating_commits": aggregated,
                 "acts_without_event": undelivered, "negative_latencies": negative})
    summary["correlation"] = corr
    print("\ncommits and sync (inside the window):")
    for k, v in corr.items():
        print(f"  {k}: {fmt(v) if isinstance(v, dict) else v}")
    if holds_during_sync == 0:
        print("  NOTE: no hold was attempted while a sync owned the hub: the SFTP-wait hypothesis is untested by this run")

    # Catalog requests announced inside the window must be served inside it
    # (by their own or a later listing): a request still pending at the quit
    # marker is work the measurement lost.
    catalog_requests = [e for e in events if e["name"] == "catalog.request" and quit_marker is not None and e["ns"] < quit_marker]
    served = [s for s in by_name.get("consumer.catalog", []) if inside(s) and s.int_attr("request") is not None]
    unserved_requests = []
    for e in catalog_requests:
        try:
            rid = int(e.get("id", ""))
        except ValueError:
            unserved_requests.append(e)
            continue
        if not any(s.int_attr("request") >= rid and s.begin > e["ns"] for s in served):
            unserved_requests.append(e)

    # ---- decision rule ---------------------------------------------------
    clicks_all = [s for s in by_name.get("click", []) if inside(s)]
    # A click that placed a control has a click.place child reporting the
    # solve epoch that publishes it; a rejected click has none.
    clicks = [s for s in clicks_all if any(c.name == "click.place" and c.int_attr("epoch") is not None for c in descendants(s))]
    landings_all = [s for s in by_name.get("landing", []) if inside(s)]
    landings = [s for s in landings_all if s.attrs.get("outcome") == "applied"]
    # Each placing click is published by a later applied landing of its pane
    # whose solve epoch is at least the one the placement reported.
    def landed(pane: str, epoch: int, after_ns: int) -> bool:
        return any(s.attrs.get("surface") == pane and (s.int_attr("epoch") or 0) >= epoch and s.begin >= after_ns
                   for s in landings)
    unlanded_clicks = [c for c in clicks if not landed(click_pane(c), place_epoch(c), c.begin)]
    # Timing covers every complete click and landing inside the window, the
    # rejected and failed ones included: a failure cannot improve the result.
    click_p95 = pct([s.inclusive_ns / 1e6 for s in clicks_all], .95) if clicks_all else float("nan")
    landing_p95 = pct([s.inclusive_ns / 1e6 for s in landings_all], .95) if landings_all else float("nan")
    forbidden = [g for g in attributed if g["gated"] and g["relevant"] and g["gap_ms"] > args.gap_ms]

    # The fixture's placements, each matched to its final landing.
    placements_total = placements_landed = 0
    unmatched_placements = []
    if args.timeline:
        with open(args.timeline, "r", encoding="utf-8") as fh:
            placements = json.load(fh).get("placements", [])
        placements_total = len(placements)
        for p in placements:
            epoch, pane, t_click = p.get("epoch"), p.get("pane"), p.get("t_click_ns")
            ok = epoch is not None and pane and t_click is not None and landed(pane, int(epoch), int(t_click))
            if ok:
                placements_landed += 1
            else:
                unmatched_placements.append({k: p.get(k) for k in ("scenario", "index", "pane", "epoch")})

    required_spans = {name: any(inside(s) for s in by_name.get(name, [])) for name in args.require_span}

    # Completeness: a measurement is rejected when it has no quit marker,
    # relevant work that began inside the window is still open at it, the
    # heartbeat did not cover it, a commit's entries were never delivered, a
    # catalog request was never served, or records were dropped or garbled.
    relevant_spans = [s for s in spans.values() if measured(s) and relevance.relevant(s)]
    unfinished = [s for s in relevant_spans if not s.complete or s.end > quit_marker]
    # Deferred consumers that the measured work scheduled but that only ran
    # after the quit marker (or never finished) are measured work the window
    # lost; shutdown's own consumers (unknown cause) are not.
    continuations = [s for s in spans.values() if quit_marker is not None and s.begin >= quit_marker and
                     any(a.name.startswith("consumer.") for a in same_thread_ancestors(s)) and
                     relevance.caused_by_measured(s, measured)]
    last_relevant_end = max((s.end for s in relevant_spans if s.complete), default=0)
    last_heartbeat = max((ts for ts, _ in heartbeats), default=0)
    dropped = [e for e in events if e["name"] == "metrics_dropped"]
    ended = [h for h in headers if h.get("end") == "1" and str(h.get("failures", "")).isdigit() and
             quit_marker is not None and h["ns"] >= quit_marker]
    recording_failures = sum(int(h["failures"]) for h in ended)
    complete = {
        "run has a quit marker": quit_marker is not None,
        "run ended with its end marker": bool(ended),
        "no recording failures": bool(ended) and recording_failures == 0,
        "no malformed or duplicate records": log.malformed == 0 and log.duplicate_ids == 0,
        "no dropped records": not dropped,
        "no failed publication": not failed_saves and not failed_landings and not acts_without_receipt,
        "no unfinished relevant span": not unfinished,
        "no measured work continued past the quit marker": not continuations,
        "heartbeat covers the last relevant span": bool(heartbeats) and last_heartbeat >= last_relevant_end,
        "every commit's entries delivered": undelivered == 0 and negative == 0,
        "every catalog request served": not unserved_requests,
        "every placing click landed": not unlanded_clicks,
    }
    if args.timeline:
        complete["every placement landed"] = placements_total > 0 and not unmatched_placements
    is_complete = all(complete.values())
    checks = {
        f"clicks that placed >= {args.min_clicks}": len(clicks) >= args.min_clicks,
        f"applied landings >= {args.min_landings}": len(landings) >= args.min_landings,
        f"holds attempted during a sync >= {args.min_overlaps}": holds_during_sync >= args.min_overlaps,
        f"click p95 < {args.click_p95_ms:.0f} ms": (click_p95 < args.click_p95_ms) if clicks_all else args.min_clicks == 0,
        f"landing p95 < {args.click_p95_ms:.0f} ms": (landing_p95 < args.click_p95_ms) if landings_all else args.min_landings == 0,
        f"no attributed gap > {args.gap_ms:.0f} ms": not forbidden,
    }
    for name, present in required_spans.items():
        checks[f"span {name} occurred"] = present
    summary["completeness"] = {"checks": complete, "unfinished": [f"{s.name}#{s.id}" for s in unfinished[:20]],
                               "continuations": [f"{s.name}#{s.id}" for s in continuations[:20]],
                               "dropped": len(dropped), "malformed": log.malformed, "duplicate_ids": log.duplicate_ids,
                               "failed_saves": len(failed_saves), "failed_landings": len(failed_landings),
                               "acts_without_receipt": len(acts_without_receipt),
                               "unserved_catalog_requests": len(unserved_requests),
                               "unlanded_clicks": [f"click#{c.id}" for c in unlanded_clicks[:20]],
                               "unmatched_placements": unmatched_placements[:20],
                               "last_heartbeat_ns": last_heartbeat, "last_relevant_end_ns": last_relevant_end,
                               "quit_ns": quit_marker, "window_start_ns": window_start}
    # Without samples there is no percentile to accept, whatever the minima.
    checks["click and landing samples present"] = bool(clicks_all) and bool(landings_all)
    verdict = "PASS" if is_complete and all(checks.values()) else "FAIL" if is_complete else "INCOMPLETE"
    summary["gate"] = {"clicks": len(clicks), "clicks_total": len(clicks_all), "click_p95_ms": click_p95,
                       "landings": len(landings), "landings_total": len(landings_all), "landing_p95_ms": landing_p95,
                       "forbidden_gaps": len(forbidden), "placements": placements_total, "placements_landed": placements_landed,
                       "checks": checks, "complete": is_complete, "verdict": verdict, "pass": verdict == "PASS"}
    print("\ncompleteness:")
    for name, ok in complete.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    print("\ndecision rule:")
    print(f"  clicks={len(clicks)}/{len(clicks_all)} p95={click_p95:.1f} ms   landings={len(landings)}/{len(landings_all)} "
          f"p95={landing_p95:.1f} ms   forbidden gaps={len(forbidden)}"
          + (f"   placements landed={placements_landed}/{placements_total}" if args.timeline else ""))
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    print("  RESULT:", verdict)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
    return {"PASS": 0, "FAIL": 1}.get(verdict, 2)


if __name__ == "__main__":
    sys.exit(main())
