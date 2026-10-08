# Six-snapshot evaluation log — 2026-10-04

- Selected resumed training `20261003_222625`, launch `20261003_222618`.
- Full existing 91-piece benchmark, manager GPU0–5, no training stop.
- Output: `/ephemeral/hendrik/fiber3d_train2/comparisons/sideband1_222625_six_20261004`.
- Independent plan review approved. Frozen current raw/EMA snapshots all step57300;
  best raw step55400, best99 step55500, best999 step57300. All selection metrics
  are dense test loss, not the independent reviewed-trace best checkpoint.
- Template SHA256: `aa5857cfbaddd8a4a8825c49ce01e4ebeddc96c30a966eabf0a4e1ea48d07e49`;
  verify unchanged before each entry. Existing prepare/validate-full completed
  for first entry; trace-series-next running best raw.
- Six requested entries completed: best raw 7,091,200 samples=19 restarts;
  current raw 7,334,400=25; best EMA99 7,104,000=20; current EMA99
  7,334,400=18; best EMA999 7,334,400=19; current EMA999 7,334,400=18.
  Each passed complete_report with 182 directions; original training root alive.
- User added best-by-restarts: froze best_reviewed_trace.pt at 6,604,800 samples,
  metric reviewed_trace/restarts_combined=14, SHA256
  `3ef5164d0d543f6a534b6e0f4aca02f1dc6dd2e6717248ace1510269c1755cd0`.
  Separate series: `sideband1_222625_best_trace_20261004`, currently evaluating.
- User requests sample counts, not updates, in reports.
- Per-directed-piece audit for best-by-loss: its previous training subset test
  had 17 restarts, not 14. Full evaluation has 18 on those same 64 directions
  and 1 outside. Three subset directions differ (-1 PHerc0139 reverse,
  +1 PHerc1299 forward, +1 PHercParis4 forward). Match used unique source,
  direction and exact reference_run metadata; all 64 matched. Cause of variation
  not established in this evaluation task; do not attribute differences only to
  adding pieces or interpret one-restart differences as definitive improvements.

## Final results (91 pieces / 182 directions each)

| Snapshot | Samples | 0139 | 0332 | 1299 | 1451 | Paris4 | Restarts | Implied mean mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw best by loss | 7,091,200 | 6 | 0 | 3 | 0 | 10 | 19 | 69.31 |
| Raw current | 7,334,400 | 10 | 0 | 5 | 0 | 10 | 25 | 53.31 |
| EMA99 best | 7,104,000 | 6 | 0 | 3 | 0 | 11 | 20 | 66.01 |
| EMA99 current | 7,334,400 | 4 | 0 | 3 | 0 | 11 | 18 | 72.96 |
| EMA999 best | 7,334,400 | 7 | 0 | 3 | 0 | 9 | 19 | 69.31 |
| EMA999 current | 7,334,400 | 6 | 0 | 3 | 0 | 9 | 18 | 72.96 |
| Raw best by restarts | 6,604,800 | 5 | 0 | 3 | 2 | 9 | 19 | 69.31 |

Implied mean is total directed reference distance (1386.175605 mm)/(restarts+1),
not equal-scroll weighting. Best-by-restarts full result includes 16 restarts
inside the 32-piece subset and 3 outside (2 PHerc1451, 1 Paris4). Original subset
was 14: one Paris4 forward direction changes 0→2. All 64 subset directions
matched by unique source/direction/reference_run metadata. Cause remains unproven;
do not claim the full-vs-subset difference is solely added pieces.

All seven runs used approved `run_fiber3d_experiment trace-series-prepare` and
`trace-series-next`, existing optimized native build and identical workload
template hash. Checkpoint hashes/selection loss values are in each series.json;
per-case audits and summaries under each output. No inference code/precision
or tracer settings changed. Training root remained alive throughout; manager
regranted GPUs0–5 after the final evaluation and workers are initializing.

