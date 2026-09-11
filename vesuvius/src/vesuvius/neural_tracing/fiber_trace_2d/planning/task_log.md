# Collection best snapshot full evaluation — 2026-09-11

- User clarified FULL50testfibers/100directions over5actualscrolls, not four-fiber short pilot.
- Starting implementation revision faa197d20 (native75013f6cc, inferencea238c4598, integration175e3645b); worktree initially clean. Previous analysis log archived in docs/saved_trace_analysis_task_log_20260911.md.
- Found existing benchmark log docs/native_trace2cp_benchmarks.md with older paul4 tests. Will link new detailed docs/benchmarks record, label differing cohorts/methods.
- best.pt embedded step17000, samples2176000, metric_name test/loss_total, metric0.3195892147353438; native output_stride4. SHA2563fecbcd6806242baa7bfc7ffcd7ec27988b28ea43e87c589e6eacc7c280a6a7c.
- Native output stride already supported in adapter/demand/shared worker: requested factor4/native4 means no second downsample of model output. Validate with tests and real full run, change only if needed.
- Saved exact classic per-scroll mean/median/P05/P95 table and command/data/snapshot/revision under docs/benchmarks/classic_full_5scroll_20260911.md, linked from existing native_trace2cp_benchmarks.md. Reproducible analysis script/JSON now include inclusive quantiles.
- Deferred GT-heading/matching investigation: GT tangent currently selects/sign-aligns the model seed direction; no minimum alignment or forced GT heading. Zero matched spans can still have physical steps and reset advances8basevox separately. No change to this behavior in snapshot comparison.
