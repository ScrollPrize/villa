# Plan: historical observation, EMA and regular training

1. Freeze reproduction findings in cffe3cde9. Inspect current and historical
   evaluator/loss, snapshot and managed RAM handoff contracts. Old metrics use
   CP-only historical JSON/control-point stream, logical batch8 and full-grid
   predictions/targets. Audit historical total-loss differences explicitly;
   share existing loss primitives rather than duplicate the trainer.
2. Add opt-in observational historical test source to the same source-tagged
   sample queue and broker. Keep current test/loss_total and angle unchanged.
   Reuse evaluator and resumable test waves for raw and EMA models. Preserve
   completed rows/logging flags across pause/resize; no synchronous new loader.
   Historical metrics do not select raw/EMA best snapshots or schedule learning.
3. Add configured EMA decays0.99 and0.999, fixed decay from initialized weights,
   FP32 shadows and foreach updates once per successful optimizer update.
   Copy nonfloating buffers; no change to BF16 forward/backward. Validate unique
   decays in(0,1). Raw checkpoint contains all EMA state/counters/best metrics;
   RAM handoff and replica verification preserve it. Legacy checkpoints may
   initialize new tracks explicitly with a startup message, never fake history.
4. Evaluate EMA on both current and historical tests using temporary weight
   copies, restore raw weights before every yielded manager boundary and in
   exception cleanup. Save independent current/best/retained EMA snapshots at
   existing snapshot boundaries; EMA best uses its current dense test loss.
   EMA exports identify decay/update count and are inference-only (reject
   accidental optimizer resume from EMA exports). Raw snapshots remain resumable.
5. CPU regressions: EMA/reference, skipped optimizer step, validation,
   snapshot/handoff restoration and raw restoration on exceptions; queue sources,
   historical definition, observational isolation and test-wave continuation.
   Approved manager-exclusive CUDA test measures EMA overhead versus raw,
   validates BF16/checkpoint inference plus pause/resize state. Short real
   pipeline test measures all six evaluations and regular stepping.
6. New config based on train_s1a_nml_16x8.json, LR.01/global128/B16 perGPU,
   max_steps2000000, regular run_path, test/checkpoint100, raw+EMA historical
   metrics. Extend approved experiment launcher for normal managed production
   start (no outer exclusive lease), record PID/log/config. Launch fresh run,
   verify calibration and actual progress/TB/snapshots, leave it running.

## Spec update
Document optional historical observation contract, fixed-decay EMA recurrence,
update frequency, snapshot export/resume rules, managed handoff and memory
accounting. Retain modern training/default metrics and locked manager protocol.

## Docs updates
Document config keys, TB tags, snapshot names, run directory and approved start/
status/stop workflow. Brief current-task log and final changelog with measured
overhead and exact CPU/GPU commands. Independent review before implementation.

## Risks and acceptance
Do not equate legacy presence loss with current loss without checking old source.
Report remaining input-backend discrepancy. Extra testing costs scale with
evaluations even when per-update EMA is cheap; report separately. Extra memory
must enter automatic calibration, not reuse an old profile. Training launch is
completion, not waiting for2M updates. No further training ablations.
