# Final EMA999 evaluation

Task: after merging evaluation-ablations, freeze final current_ema_0p999.pt from stopped 5scroll_json_eligible_128_sd1_16x8_ema_20260914_105436 training; evaluate all 50 fibers across five scrolls with visualization.

1. Freeze and record SHA256, update/sample metadata; pin configuration to immutable snapshot.
2. Reuse approved evaluation runner: prepare, validate-full, full. All eight GPUs exclusively via manager, existing parameter-keyed inference cache. Same eligible spans, 40 thickness / 80 width ellipse, 160 excursion grace, 80 reference-arc restart filtering, embedding off.
3. Monitor through completed evaluation and all renders; inspect images and per-fiber index. Compare global/per-scroll restarts, mean, median and P05/P95 against same-protocol Classic and previous EMA snapshots. Mean = total directed reference distance / (reported restarts + 1).
4. Persist benchmark report, provenance, command and completion log.

## Spec update
No algorithm/protocol changes. Preserve merged visualization and cache behavior.

## Docs updates
Record final result under docs/benchmarks, changelog and this dedicated task log, leaving merge-protected task files intact.

## Validation
Independent plan review, prepare/validate-full, 180 eligible directional cases, complete render metadata and per-fiber index. Compare coverage with previous 6867.627253 mm. No training or exhaustive sweep.
