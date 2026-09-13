# Embedding tuning plan

1. Freeze raw4300/550400, same hard held-out fiber and bidirectional metric. Run off/.25 controls and signed embedding cache under manager nonexclusive oneGPU20GiB/16CPU. Training stays active. Restarts ranked by existing total_failures, complete traces only.
2. Generalize existing sweep configuration, not tracer implementation: explicit candidate histories accepted; weight search accepts selected history and starting weight instead of hardcoded old optimum. Keep old defaults/tests. Validate values and preserve all non-embedding trace args.
3. First grid weight.25: spacing8/32/128 and window64/256/1024/4096 with count64 and uniform/linear/Gaussian (skip spacing>window); include old controls. This covers short and newly relevant long ranges. Select minimum failures, then .05 weight neighbor descent from.25 with that history (ties retain center). Compare count8/16/32/64 at winning weight/history, and a local history refinement at the new weight if weight changed. Retain zero-weight control as candidate. Report local, not global optimum. Stop refining when no strict improvement among tested neighbors.
4. Reuse cached inference; manager fills only additional missing coverage. Require exact old-control replay and exact repeat of final winner. CPU regression tests via existing approved test runner, then complete actual tuning. Report settings/counts/restarts relative to fresh off/default and log full result directories. No global default changes unless separately requested.

## Spec update
No numerical, training or protocol change. Configurable experiment search only; settings restricted to embedding parameters, unchanged deterministic selection and fail-closed coverage.

## Docs updates
New tuning report with snapshot/data, search grid and local refinement, complete controls/repeat, timings and single-fiber overfit limitation. Task log/status/changelog brief.
