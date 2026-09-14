# Aggregate restart interval mean — 2026-09-14

User requests combined directed distance/(reported restarts+1), including across multiple fibers/directions. Old divisor added one per directed run. Raw segments/restart events/quantiles unchanged. Prior log archived in docs/benchmarks/eligible_ema99_u5200_task_log_20260914.md. No inference/tracing rerun.

Shared segment_statistics mean now uses denominator failures+1; separate
mean_definition/mean_denominator preserve actual reliability segment counts.
correct_summary updates mm/base-voxel/percent means; comparison invariant and
visualization aggregate means/formulas updated. Single directed case unchanged.
Independent review approved. Tests21native+138Python passed46.13s via approved
run_fiber3d_native_replay_tests. git diff --check clean. Saved-summary correction
idempotence validated on all three real models; no original summaries/traces
overwritten. Corrected outputs in docs/benchmarks/combined_restart_mean_20260914.md
and.json; prior table mean definitions marked superseded.
0332 EMA:425.160903676/3=141.720301225mm. Global means Classic23.439001,
EMA1800 39.243584,EMA5200 46.091458; normalized scores23.401875,54.904589,
55.845764 respectively. Raw count/length/quantile values unchanged. No deviations.
