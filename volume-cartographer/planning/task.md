# Task: finish remote byte-stream bandwidth adaptation

Correct the incomplete streamed-bandwidth implementation.

- Remote measurement starts when the HTTP chunk request is issued, so request
  latency and time to first byte are included.
- Received response-body bytes are the sole remote bandwidth numerator.
- Intervals with no remote request in flight remain excluded.
- The HTTP path must not use the old `admission * 4` completion window.
- Remove obsolete completion-window benchmark controls and diagnostics.
- Keep successful completions only for latency, failure, and the requirement
  that an adaptive epoch observe at least one completion per admitted worker.
- Local and custom fetches must not update displayed remote bandwidth, adaptive
  admission history, or persisted remote state.
- Preserve the requested fourfold initial concurrency probe and twofold
  refinement probes.
