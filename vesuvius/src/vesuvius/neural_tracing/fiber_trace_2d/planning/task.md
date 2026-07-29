# Python Native 3D Trace2CP Multi-Fiber Input

Update the Python native 3D Trace2CP tool so `--fiber-json` accepts multiple
JSON fibers. In that mode the tool should:

- run whole-fiber tracing over the supplied JSON files sequentially;
- reuse one loaded model across the per-fiber runs;
- report an accumulated restart-rate score over all fibers;
- keep existing single-fiber and sample-index behavior unchanged;
- write indexed per-fiber summaries and indexed per-fiber visualizations when
  `--vis` is enabled, so each JSON gets its own output.
