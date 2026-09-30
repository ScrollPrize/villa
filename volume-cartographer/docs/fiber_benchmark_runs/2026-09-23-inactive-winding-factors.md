# Zero-weight winding-factor correction

## Scope

The reduced hard-perpendicular / parallel-one policy previously retained
zero-contribution winding measurements and their BP message buffers. This
correction removes identically neutral measurements before winding graph
preparation, so they cannot affect messages, components, gauges, initialization
or defect incidence. Original-index diagnostic records remain available.
Continuity, independently enabled signs and local-span penalties remain active.
Fixed-scale classification uses the actual effective target. Adaptive calibration
removes a factor only if it is inactive across all reachable scale classes;
an exact-zero target cannot reactivate via unrelated class weights.

Scope: this fixes the joint-grid BP solver used by these benchmarks. Other
legacy plain/alternating winding solvers are not changed. The orientation
prepass remains independent: zero winding magnitude is not a request to remove
its nonzero direction-class evidence, nor an independently enabled sign loss.
Canonical joint-orientation inputs have complementary nonzero total orientation
scores, so they are not fully neutral merely because winding weights are zero.

No predictor, trace generation, scientific weight, iteration limit, reference
protocol or inference cache changes. Solver outputs can change because the old
zero-cost factors incorrectly influenced finite-iteration behavior; this is not
a claim of bit-identical speed-only optimization.

## Bug found during validation

The first trial removed neutral edges only after preparation. It ran the small
control in52.8636s versus68.6863s, with10719/41833edges removed. All initial factor
diagnostics and decoded energy matched exactly, but conditioned results changed
from23/26to24/26exact references. That trial was not accepted as an equivalent
speed-only change and did not launch the large solve.

Investigation found that support expansion filled new log-message entries with0
while preserving negative normalized old entries. Damping then made a neutral
factor temporarily prefer new states. Preserving such slots would preserve the
bug. The user explicitly approved correcting the behavior, with zero-weight
input required to match physical omission. Tests must cover inactive bridges,
short iteration caps, defects and support expansion, not only converged scores.

## Reproduce

Build using the existing no-install runner:

```sh
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
```

Native regression suite, then small control, then large saved-trace solve:

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_winding_tests_20260923.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_beam_bp_perpsign_zero_fixed_20260923.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_beam2048_perpsign_zero_fixed_20260923.json
```

RelWithDebInfo, 128 requested CPU workers, one execution per configuration.
Runner saves exact command, source patch, input/reference/binary hashes and
wall/user/system time plus peak RSS. No GPU. Existing traces reused.

## Baselines and validation

- Small 1024³ control: 68.6863 s whole command, 23 exact / 3 wrong / 0 missing
  references (88.4615%); constraint score −119.5721% (23081 / 19303).
- Large 2048³ interrupted run: initial 500-message solve 349.0 s,
  16,385,276 candidate/message states. Interrupted at conditioned solve start;
  no completed end-to-end baseline. Generated 3,697 traces are retained.
- Python runner tests: 7 passed.
- Native winding regressions:100passed, including exact omission equivalence at
  caps1/2/7/100, inactive bridges, defects, all-scale neutrality, active-sign
  retention and uniform/log-gauge-invariant support expansion. Independent
  implementation review passed.

## Corrected small control

| Measurement | Before | Corrected |
| --- | ---: | ---: |
| Whole command wall | 68.6863 s | 53.0352 s |
| Initial winding solve | 13.1 s | 9.8 s |
| Initial message/state count | 516,249 | 385,926 |
| Initial BP edges | 41,833 | 31,114 |
| Reference exact | 88.4615% | 88.4615% |
| Constraint score | −119.57% | −118.73% |
| Peak RSS | 582,476 KiB | 580,872 KiB |

One execution per version, same saved traces/parameters,128requested workers.
Whole-command wall is22.8% lower;25.6%of initial BP edges are removed. This
includes a semantic bug correction, not a strictly bit-identical optimization.
All41,833initial diagnostic CSV rows nevertheless match exactly. Both corrected
and old initial/reference solves hit the unchanged500-message limit.

Corrected constraint counts23007problematic/19377fulfilled; round0has
23exact/3wrong/0missing. The three incorrect gauge-aligned estimates remain
8→7,11.5→12.5,12→8. Raw gauge offsets and conditioned closure can change;
reference accuracy alone does not imply every solver output is unchanged.
Intermediate post-preparation compaction trial24/26is superseded and is not
reported as the corrected result.

Corrected artifact prefix:
`/ephemeral/hendrik/fiber_crop_bench/20260923_v1/bp-beam-new-perpsign-p1low-zero-fixed`.
The larger rerun completed under the corresponding2048output root. Its initial
500-iteration solve completed in276.2s versus349.0s before (20.9% lower). Initial
edges1358559→978770 (379789inactive removed;28.0%), candidate/message states
16385276→11828403. Whole corrected command1492.2988s, peakRSS5925824KiB,
returncode0. Reference result21/26=80.7692%,5wrong/0missing; constraint score
−210.4353%(949097/451016). Larger context did not improve the corrected small
control. Initial, conditioned and fresh solves reached the unchanged iteration
cap. No old complete large-run wall time exists because that run was explicitly
interrupted. Full assignments and factor counts are in the linked2048report.

The larger-crop scientific comparison is recorded separately in
[2048³ context experiment](2026-09-23-beam2048-context.md).
