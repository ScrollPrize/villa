#!/usr/bin/env python3
"""Dependency-free parser for passive Valgrind scheduler and DRD traces."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

FUTEX_COMMAND_MASK = 0x7F
FUTEX_WAIT_COMMANDS = {0, 9}
FUTEX_WAKE_COMMANDS = {1}
SCHED_RE = re.compile(r"SCHED\[(\d+)\]:\s+(.*)")
FUTEX_RE = re.compile(
    r"SYSCALL\[\d+,(\d+)\]\(202\) sys_futex \( "
    r"(0x[0-9a-fA-F]+), (\d+), (\d+),"
)
FUTEX_RESULT_RE = re.compile(
    r"SYSCALL\[\d+,(\d+)\]\(202\) \.\.\. \[async\] --> "
    r"(Success|Failure)\(0x([0-9a-fA-F]+)\)"
)
CLONE_RE = re.compile(r"SYSCALL\[\d+,(\d+)\]\(56\) sys_clone")
SET_TID_ADDRESS_RE = re.compile(
    r"SYSCALL\[\d+,(\d+)\]\(218\) sys_set_tid_address \( (0x[0-9a-fA-F]+) \)"
)
DRD_SEGMENT_RE = re.compile(r"New segment for thread (\d+) with vc \[([^]]*)\]")
DRD_CLOCK_RE = re.compile(r"(\d+):\s*(\d+)")


@dataclass
class TraceEvent:
    sequence: int
    thread: int
    kind: str
    detail: dict[str, object] = field(default_factory=dict)
    dependencies: list[tuple[int, str]] = field(default_factory=list)
    duration: float = 0.0

    def as_json(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "thread": self.thread,
            "kind": self.kind,
            "detail": self.detail,
            "dependencies": [
                {"sequence": sequence, "kind": kind}
                for sequence, kind in self.dependencies
            ],
        }


@dataclass
class PendingWait:
    address: str
    begin_sequence: int
    blocked: bool = False
    resume_sequence: int | None = None


@dataclass
class WakeToken:
    sequence: int
    remaining: int


@dataclass
class ParsedTrace:
    events: list[TraceEvent]
    full_quanta: dict[int, int]
    blocking_waits: int
    matched_waits: int
    unmatched_waits: int
    nonfutex_waitsys: int
    happens_before_edges: int = 0
    unresolved_happens_before: int = 0


def _append_event(
    events: list[TraceEvent], thread: int, kind: str, **detail: object
) -> TraceEvent:
    event = TraceEvent(len(events), thread, kind, detail)
    events.append(event)
    return event


def parse_core_trace(path: Path) -> ParsedTrace:
    events: list[TraceEvent] = []
    pending: dict[int, PendingWait] = {}
    wake_tokens: dict[str, list[WakeToken]] = {}
    full_quanta: dict[int, int] = {}
    blocking_waits = 0
    matched_waits = 0
    nonfutex_waitsys = 0
    tid_addresses: dict[str, int] = {}

    for line in path.read_text(errors="replace").splitlines():
        clone = CLONE_RE.search(line)
        if clone:
            _append_event(events, int(clone.group(1)), "thread_create")

        tid_address = SET_TID_ADDRESS_RE.search(line)
        if tid_address:
            tid_addresses[tid_address.group(2).lower()] = int(tid_address.group(1))

        futex = FUTEX_RE.search(line)
        if futex:
            thread = int(futex.group(1))
            address = futex.group(2).lower()
            operation = int(futex.group(3))
            argument = int(futex.group(4))
            command = operation & FUTEX_COMMAND_MASK
            if command in FUTEX_WAIT_COMMANDS:
                if thread in pending:
                    raise RuntimeError(f"thread {thread} starts nested futex waits")
                event = _append_event(
                    events,
                    thread,
                    "futex_wait",
                    address=address,
                    operation=operation,
                    expected=argument,
                    blocked=False,
                )
                pending[thread] = PendingWait(address, event.sequence)
            elif command in FUTEX_WAKE_COMMANDS:
                event = _append_event(
                    events,
                    thread,
                    "futex_wake",
                    address=address,
                    operation=operation,
                    requested=argument,
                )
                wake_tokens.setdefault(address, []).append(
                    WakeToken(event.sequence, argument)
                )

        result = FUTEX_RESULT_RE.search(line)
        if result:
            thread = int(result.group(1))
            wait = pending.pop(thread, None)
            if wait is not None:
                value = int(result.group(3), 16)
                if wait.resume_sequence is None:
                    resume = _append_event(
                        events,
                        thread,
                        "futex_resume",
                        address=wait.address,
                        blocked=wait.blocked,
                    )
                    wait.resume_sequence = resume.sequence
                resume = events[wait.resume_sequence]
                resume.detail["result"] = result.group(2).lower()
                resume.detail["result_value"] = value

        for sched in SCHED_RE.finditer(line):
            thread = int(sched.group(1))
            action = sched.group(2)
            if action.startswith("entering VG_(scheduler)"):
                _append_event(events, thread, "thread_start")
            elif "VG_(scheduler):timeslice" in action and action.startswith(
                "releasing lock"
            ):
                _append_event(events, thread, "work_quantum")
                full_quanta[thread] = full_quanta.get(thread, 0) + 1
            elif action.startswith("releasing lock") and "VgTs_WaitSys" in action:
                wait = pending.get(thread)
                if wait is not None:
                    wait.blocked = True
                    events[wait.begin_sequence].detail["blocked"] = True
                    blocking_waits += 1
                else:
                    nonfutex_waitsys += 1
            elif action.startswith("acquired lock") and "client_syscall" in action:
                wait = pending.get(thread)
                if wait is not None and wait.resume_sequence is None:
                    resume = _append_event(
                        events,
                        thread,
                        "futex_resume",
                        address=wait.address,
                        blocked=wait.blocked,
                    )
                    wait.resume_sequence = resume.sequence
                    if wait.blocked:
                        for token in wake_tokens.get(wait.address, []):
                            if (
                                token.sequence > wait.begin_sequence
                                and token.remaining > 0
                            ):
                                resume.dependencies.append(
                                    (token.sequence, "futex_wake")
                                )
                                token.remaining -= 1
                                matched_waits += 1
                                break
            elif action.startswith("exiting VG_(scheduler)"):
                _append_event(events, thread, "thread_finish")

    if pending:
        raise RuntimeError(f"unfinished futex syscalls for threads {sorted(pending)}")

    finishes = {
        event.thread: event.sequence
        for event in events
        if event.kind == "thread_finish"
    }
    previous_by_thread: dict[int, TraceEvent] = {}
    for event in events:
        if (
            event.kind == "futex_resume"
            and bool(event.detail.get("blocked"))
            and not any(kind == "futex_wake" for _, kind in event.dependencies)
        ):
            joined_thread = tid_addresses.get(str(event.detail["address"]))
            finish = finishes.get(joined_thread) if joined_thread is not None else None
            previous = previous_by_thread.get(event.thread)
            if (
                finish is None
                and previous is not None
                and previous.kind == "futex_wait"
            ):
                prior_finishes = [
                    candidate
                    for candidate in events[previous.sequence + 1 : event.sequence]
                    if candidate.kind == "thread_finish"
                ]
                if prior_finishes:
                    lifecycle = prior_finishes[-1]
                    joined_thread = lifecycle.thread
                    finish = lifecycle.sequence
            if finish is not None and finish < event.sequence:
                event.dependencies.append((finish, "thread_finish"))
                event.detail["joined_thread"] = joined_thread
                matched_waits += 1
        previous_by_thread[event.thread] = event

    unmatched_waits = blocking_waits - matched_waits
    _add_program_and_lifecycle_dependencies(events)
    return ParsedTrace(
        events,
        full_quanta,
        blocking_waits,
        matched_waits,
        unmatched_waits,
        nonfutex_waitsys,
    )


def parse_drd_trace(path: Path) -> ParsedTrace:
    """Parse DRD vector clocks and scheduler quanta into one dependency graph."""
    core = parse_core_trace(path)
    events: list[TraceEvent] = []
    full_quanta: dict[int, int] = {}
    segments: dict[tuple[int, int], int] = {}
    unresolved: list[tuple[TraceEvent, int, int]] = []
    previous_clock: dict[int, dict[int, int]] = {}

    for line in path.read_text(errors="replace").splitlines():
        segment = DRD_SEGMENT_RE.search(line)
        if segment:
            thread = int(segment.group(1))
            clock = {
                int(owner): int(value)
                for owner, value in DRD_CLOCK_RE.findall(segment.group(2))
            }
            own_clock = clock.get(thread)
            if own_clock is None:
                raise RuntimeError(f"DRD segment for thread {thread} has no own clock")
            event = _append_event(
                events,
                thread,
                "hb_segment",
                clock=clock,
                own_clock=own_clock,
            )
            for owner, value in clock.items():
                if owner != thread and value > previous_clock.get(thread, {}).get(
                    owner, 0
                ):
                    unresolved.append((event, owner, value))
            key = (thread, own_clock)
            if key in segments:
                raise RuntimeError(f"duplicate DRD vector-clock segment {key}")
            segments[key] = event.sequence
            previous_clock[thread] = clock

        for sched in SCHED_RE.finditer(line):
            thread = int(sched.group(1))
            action = sched.group(2)
            if action.startswith("entering VG_(scheduler)"):
                _append_event(events, thread, "thread_start")
            elif "VG_(scheduler):timeslice" in action and action.startswith(
                "releasing lock"
            ):
                _append_event(events, thread, "work_quantum")
                full_quanta[thread] = full_quanta.get(thread, 0) + 1
            elif action.startswith("exiting VG_(scheduler)"):
                _append_event(events, thread, "thread_finish")

    unresolved_count = 0
    happens_before_edges = 0
    for event, owner, value in unresolved:
        predecessor = segments.get((owner, value))
        if predecessor is None or predecessor >= event.sequence:
            unresolved_count += 1
            continue
        event.dependencies.append((predecessor, "drd_happens_before"))
        happens_before_edges += 1

    _add_program_and_lifecycle_dependencies(events)
    return ParsedTrace(
        events=events,
        full_quanta=full_quanta,
        blocking_waits=core.blocking_waits,
        matched_waits=core.matched_waits,
        unmatched_waits=core.unmatched_waits,
        nonfutex_waitsys=core.nonfutex_waitsys,
        happens_before_edges=happens_before_edges,
        unresolved_happens_before=unresolved_count,
    )


def _add_program_and_lifecycle_dependencies(events: list[TraceEvent]) -> None:
    previous_by_thread: dict[int, int] = {}
    latest_create: int | None = None
    for event in events:
        previous = previous_by_thread.get(event.thread)
        if previous is not None:
            event.dependencies.append((previous, "program_order"))
        if event.kind == "thread_create":
            latest_create = event.sequence
        elif (
            event.kind == "thread_start"
            and event.thread != 1
            and latest_create is not None
        ):
            event.dependencies.append((latest_create, "thread_create"))
        previous_by_thread[event.thread] = event.sequence


def write_event_stream(path: Path, events: list[TraceEvent]) -> None:
    with path.open("w") as stream:
        for event in events:
            stream.write(json.dumps(event.as_json(), sort_keys=True) + "\n")


def read_event_stream(path: Path) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for line in path.read_text().splitlines():
        value = json.loads(line)
        event = TraceEvent(
            sequence=int(value["sequence"]),
            thread=int(value["thread"]),
            kind=str(value["kind"]),
            detail=dict(value.get("detail", {})),
            dependencies=[
                (int(dependency["sequence"]), str(dependency["kind"]))
                for dependency in value.get("dependencies", [])
            ],
        )
        if event.sequence != len(events):
            raise RuntimeError(f"non-contiguous event sequence in {path}")
        events.append(event)
    return events
