# Saved full-test trace-length analysis — 2026-09-11

- Completed code/logs committed as 75013f6cc (native), a238c4598 (Lasagna), 175e3645b (Vesuvius).
- Analyze completed full cap8 heap run only, no rerun. Previous log archived in docs/dynamic_inference_batching_task_log_20260911.md.
- Existing mean_segment_length_mm uses the intentional whole-corpus reliability convention length/(failures+1); do not change it or mistake it for the empirical mean failure-ended span. Use stored traced_span_base_voxels for requested before-restart metric and distinguish terminal spans.
- Independent plan/metric review passed and independently reproduced all six table rows.
- Validated 100 complete saved case artifacts against summary, every failure span, provenance pairing, segment count and credited reference length. No inference/tracing rerun. Global mean before restart 3.427927322 mm across1995 failures (65 zero spans); including100 terminal segments 3.855689840 mm. Historical corpus reliability remains4.048876783 mm, intentionally different.
- Per actual scroll failed-span mean mm: PHerc0139 2.232873; PHerc0332 12.994266; PHerc1299 3.417562; PHerc1451 2.341021; Paris4 pooled3.785202. Reference-progress metric, not traced-polyline length (global latter3.493324400 mm). Both Paris4 collections retained separately in JSON.
- Saved docs/trace_restart_analysis_20260911.md/.json/.py. Script is the exact executed read-only standard-library calculation with arithmetic/artifact assertions. No runtime/spec changes, no omitted requested work.
- Existing profile confirms sustained inference bottleneck (~82tiles256³/s across8GPUs, ready inputs full, peak publication backlog11); sparse tail not necessarily GPU-bound. Full cap8 wall516.699s vs batch1 heap860.380s.
