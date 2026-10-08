# Current EMA99 matched full evaluation — 2026-09-14 13:45 UTC

Fresh current EMA99 from5scroll_json_eligible_128_sd1_16x8_ema_20260914_105436. Normal40/tangent80/grace160,episode80,no rendering/embedding. Prior log archived as docs/benchmarks/thickness40_grace160_task_log_20260914.md. No training/compression changes.

Frozen current EMA99 update5200/samples665600/test_ema_0p99 loss0.3008716558.
SHA256 ccf706ab68d51c76910a8134b6970c3289a1e2503320d74c8eb75fa1f74209a6.
Snapshot and metadata under /ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema99_current_20260914_1345/frozen and snapshot.json.
Config eval_eligible_ema99_current_20260914_1345.json pins immutable checkpoint
and SHA. Independent review approved unchanged policy; workspace build no work.

21native+136Python tests passed48.63s. prepare/validate-full confirmed frozen
checkpoint/split/normal hashes. Native reference plan:50fibers/90runs/180directions,
no excluded-only fibers. Full managed task0bbd121d-aa42-4ced-8222-4fdf4cee02f2
granted all8GPUs/64logicalCPU. While running user requested previous two-model
full per-scroll/current-policy comparison; reported shared-helper scores:
Classic-normalized14.853169→23.142353; equal-scroll mean14.853169→24.403490;
pooled14.550058→19.400077. No strict-policy results mixed into this comparison.

Full evaluation completed180/180; managed runner exited0. New model inferred
29225tiles and published9150chunks, zero warm inference chunks. CT/normal cache
reused. Coordinator wall352.239569s; observed97–100% GPU utilization.
New EMA99u5200:148restarts,20.937888mm pooled mean,8.202461median,
P05.143040/P9588.437892mm. Previousu1800:174/19.400077; Classic292/14.550058.
Classic-normalized equal-scroll score23.248342 vs23.142353previous (+0.46%);
arithmetic scroll mean24.523323 vs24.403490. Paris4/0139 improve,1299
regresses8→11restarts;0332/1451 counts unchanged. Matched coverage verified
6867.627252833mm and180directions for all three. Full per-scroll statistics,
combined scores, provenance and reusable commands saved in
docs/benchmarks/eligible_ema99_u5200_20260914.md. No implementation changes,
no deviations; no visualization requested/performed.
