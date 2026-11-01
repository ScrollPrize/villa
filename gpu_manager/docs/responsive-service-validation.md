# Responsive manager validation — 2026-09-11

## Cause and change

The old socket loop ran descendant discovery for each tracked process, repeatedly
rebuilding the complete host process map. With 152 tracked identities, three
unprofiled scans took 5.213/5.406/5.731 seconds, exceeding the client's five-second
RPC timeout. A profiled scan took 6.613 seconds with 152 parent-map constructions.

The production service now has separate OS processes for public socket I/O,
state/policy/journal ownership, and asynchronous process collection. The collector
reads one parent/identity table per scan. Snapshot-proven alive identities avoid
repeated inspection during scans; missing/unknown/dead evidence is verified
directly, as are client release reports. Poll/status avoid transaction copies and
poll avoids redundant reconciliation. No wire protocol or allocation change.

## Measurements

On this host, three collections of 2,746 process rows took
68.04/65.81/61.81 ms (Python 3.14, existing venv; no compilation). This measures
collection only, not full policy processing, and should not be presented as a
full scan speedup against the old end-to-end times.

A separate end-to-end measurement on an in-memory copy of the live journal,
including collection, snapshot construction and `Engine.scan_processes()`, took
74.36/71.29/67.79 ms with 2,756 host processes and 153 tracked identities. The old
baseline had 152 tracked identities. Neither measurement includes journal fsync;
the new measurement did not modify the live journal or allocations.

In a synthetic-inventory service test with a deliberately two-second collector
delay, 30 sequential status+poll pairs took min/median/max
10.4/10.5/10.9 ms. This validates replies, not merely socket accepts.

## Reproduction and safety

From the Villa root:

```bash
python -m pytest -q gpu_manager/tests
```

`test_process_service.py -s` prints collection and RPC measurements. Its five
tests cover real process IPC, slow collection, conservative snapshot identity
handling, journal recovery, and singleton retention after public-parent hard
death. The crash test kills only processes created by that test. No GPUs are
initialized and no live service or allocation is changed.
The complete manager suite passed: **122 tests in 31.16 seconds**.

Mutation replies remain ordered after durable journal writes. A stalled disk or
long policy computation can still delay actual replies in the state owner; no
unpersisted grant is acknowledged to conceal this. Production-load validation
after a controlled manager restart remains necessary. The live manager has not
been restarted by this work.
