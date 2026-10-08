# Issues

## Resolved: on-demand fiberlet generation used one core

During replay, rows such as

```text
fiber_replay_cache_chunk stage=fiberlets status=started key=107,36,44 inputs=116582 outputs=0 elapsed_seconds=0
```

reliably correspond to approximately one core of CPU load. The on-demand path
generates only one fiberlet owner chunk at a time, and the expensive candidate
preparation/DP work inside that generation is effectively single-threaded.
This is distinct from waiting for anchor data or storage I/O.

Resolved on 2026-08-20. Candidate generation, preparation, sampling, DP search,
and prepared-geometry release now use the configured worker count within one
owner chunk. Results are merged in canonical source order, and serial versus
parallel tests require byte-identical reports and geometry. On the measured
Paris4 chunk `107,34,45` (105,730 input anchors and 63,932 output fiberlets),
wall time fell from 34.795 s to 4.344 s while the stored prefix and route files
remained byte-identical.

## Resolved: replay output had no globally consistent progress

The current progress output interleaves incompatible counters:

- Greedy `step=N/M` is local to one restart segment. It resets to zero after a
  failure and `M` is recomputed from the remaining reference length multiplied
  by the safety factor.
- Greedy failure `reference_arc_fraction` is global to the selected reference
  interval.
- Anchor and fiberlet rows expose only spatial chunk keys and per-chunk input /
  output counts. They provide no location or progress along the selected
  reference interval and no completed/required global work count.
- The two evaluators run concurrently, so these unrelated rows are interleaved.

Consequently the displayed progress jumps backward and does not answer how far
the replay or its required preprocessing has advanced.

Resolved on 2026-08-20. Both evaluators now report monotone global reference arc
and fraction, and greedy restart-local counters are explicitly named
`local_step` and `local_budget`. Cache rows include a stable schedule index,
scheduled and generated chunk counts, nearest global reference location, and
internal phase progress. Evaluators emit explicit terminal rows, and running
progress no longer presents `trace_distance_not_reached` as a global status.
