# Complete-graph fixed-reference neighborhood diagnostic — 2026-09-28

## Scope and reproducibility

Previous work checkpoint: `a185b0768`. Largest-component filtering is now off by default for BP-only evaluation. Historical filtering requires explicit `--winding-largest-component-only`; none of these runs uses it.

Frozen 2048 Paris4 beam8 crop, 40/80 coverage ellipse, 45-degree suppression and forward/backward consistency:
`/ephemeral/hendrik/fiber_crop_bench/20260923_beam2048_v1/beam-crop-new2048-coverage40x80-angle45-reverse-v2.zarr`.

Tracing inference: EMA999 **best u22000**, training `5scroll_json_eligible_refinement_128_sd1_16x8_ema_20260915_212757`, prediction `7ff0ce6c`; snapshot SHA256 `67edd3a1bde162c7105ee37d849a0f0b60a9b840dd37fc7ff6e17e0c3a2e9742`. This is not the later full-resolution model. No inference or retracing was performed.

References: 26 tagged `hendrik_crop1` fibers in `/home/hendrik/test_datasets/2026-09-01_fiber_stack2`, represented by 220 fixed pieces. Normal manifest and all input hashes are recorded in the per-run provenance files.

Protocol: source-certified local constraint events, midpoint splitting without overlap, parallel factors omitted after common segmentation, hard perpendicular signs, joint ordinary H/V/winding/defect states, finite untruncated physical-winding L1 continuation, fixed phase0.5/scale0.822, 64 workers/run, 500-message cap, no decoder repair. All runs reached that cap; per user instruction no iteration-limit extensions were attempted. No claim of global optimality is made.

D is the existing incident-evidence-scaled piece disablement coefficient; C is the multiplier of absolute physical winding step on each continuation. D is not a uniform flat price per piece.

## Selection and fixed-reference audit

N2 seeds an ordinary source from an orthogonal reference cross-constraint with unquantized `abs(signed distance)*0.822 <= 2`. Keep all pieces of each selected source, all induced ordinary factors and all its reference factors, including those beyond the selection radius. No transitive ordinary-neighbor flood.

- All: 2,778 ordinary fibers, 68,851 ordinary pieces, 104,743 combined factors.
- N2: 548 fibers (**19.73%**), 29,597 ordinary pieces (**42.99%**), 45,157 combined factors.
- All 220 reference pieces remain active with their exact fixed H/V, integer winding, phase sign and physical coordinate in every run.
- Canonical even-H/odd-V reference gauge: global sign -1. Normal component0 has113 units agreeing with negative sign and0 with positive sign. No cross normal component lacks reference sign evidence.
- N2 has one anchored integer-gauge component; all has666 components,665 without references. These disconnected components are retained, not silently removed. Their absolute offsets cannot be assessed against the reference ladder.

## Results

Disabled percentages use ordinary pieces/union-length. Jump percentages use active-active ordinary continuation edges; defect-incident boundaries are counted separately, never as zero or nonzero jumps.

| Solve / scored cohort | D | C | Disabled pieces | Disabled length | Nonzero jumps / active edges | Disabled runs | Jump runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| N2 / N2 | 10 | 10 | 2083 (7.04%) | 10.32% | 326/25352 (1.29%) | 1795 | 320 |
| All / all | 10 | 10 | 4603 (6.69%) | 8.70% | 820/57703 (1.42%) | 4195 | 806 |
| N2 / N2 | 3 | 10 | 2090 (7.06%) | 10.34% | 427/25343 (1.68%) | 1796 | 421 |
| N2 / N2 | 10 | 3 | 1271 (4.29%) | 6.88% | 1912/26918 (7.10%) | 1015 | 1783 |
| All / **same N2 fibers** | 10 | 10 | 2563 (8.66%) | 12.28% | 343/24454 (1.40%) | 2245 | 335 |

The last row is derived by original source-trace ID from the all-fiber solve, scoring precisely the same548 fibers/29,597 pieces as N2. This avoids conflating a changed scoring cohort with the effect of graph restriction.

### Interpretation

- Restriction reduces disablements and slightly reduces jumps on the matched cohort (7.04% versus8.66% disabled pieces;1.29% versus1.40% jump edges).
- Cheaper continuation steps (C3) trade fewer defects for many more jumps. Cheaper disablement (D3) does not improve either of those two counts in this small trial. Different cost settings have different objectives, so raw energies are not comparable across rows.
- N2 D10/C10:2,083 disabled pieces form1,795 runs (1.16 pieces/run);326 jump edges form320 connected runs. This is mostly isolated events rather than long contiguous runs, though471/548 fibers (85.95%) have at least one defect or jump. No conclusion about geometric error correlation follows without ordinary ground truth.
- N2 D10/C10 hard perpendicular sign infringements are0 among active ordinary and reference-cross factors. This does **not** mean all evidence is satisfied:1,480/2,861 reference-cross factors (51.73%) touch a disabled piece;414/1,381 active reference-cross factors (29.98%) disagree in H/V. Ordinary active H/V disagreement is861/10,837 (7.95%). The exported magnitude agreement classes are diagnostic; their weights are zero in this sign-focused protocol.
- Reference accuracy would be100% by construction, not a meaningful success score. Jumps/disablements are diagnostics of the fitted ordinary graph, not established geometric ground-truth errors.
- The D10/C10 pilot and final rerun produced identical solution counts/energies. Export-only additions did not change the solve.

## Runtime and verification

RelWithDebInfo native build. One final repetition per configuration; four64-worker processes ran concurrently. These are runtime observations, not controlled speedup measurements.

| Run | Solver seconds | Whole command seconds |
|---|---:|---:|
| n2-d10-c10 | 38.72 | 97.81 |
| nall-d10-c10 | 77.10 | 136.80 |
| n2-d3-c10 | 36.18 | 94.79 |
| n2-d10-c3 | 34.80 | 93.13 |

Tests:123 native winding cases passed, including inclusive calibrated radius, whole-source retention of disconnected pieces, no neighbor flooding, missing measurement handling, orthogonal-only selection, all-control and invalid-input checks.26 Python crop/sweep/report tests passed. Independent plan/code review completed.

## Commands and artifacts

Build:

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
```

Each of the four configurations can be passed to the existing approved runner. Example:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_reference_neighborhood_n2-d10-c10_20260928.json
```

Configuration suffixes: `n2-d10-c10`, `nall-d10-c10`, `n2-d3-c10`, `n2-d10-c3`.

Native tests:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_local_events_tests_20260927.json
```

Python tests:

```bash
/home/hendrik/.venv_las/bin/python -m pytest -q -o 'pythonpath=/home/hendrik/villa2/vesuvius/src /home/hendrik/villa2' vesuvius/tests/neural_tracing/test_crop_benchmark.py vesuvius/tests/neural_tracing/test_local_cost_sweep.py vesuvius/tests/neural_tracing/test_local_constraint_report.py vesuvius/tests/neural_tracing/test_density_penalty_sweep.py
```

Data directory: `/ephemeral/hendrik/fiber_crop_bench/20260928_reference_neighborhood_v1/`.

For each prefix:
- `_reference_neighborhood.json`: full summary, fixed-reference sign audit, per-source/per-piece/per-edge results, per-class agreement split into reference-only/cross/ordinary groups.
- `_reference_neighborhood_pieces.csv`, `_reference_neighborhood_continuations.csv`: original artifact source IDs and ordinary piece IDs, arc bounds, active flags, winding coordinates/jumps.
- `.run.json`, `.source.patch`, `.log`, `.resources.txt`: exact binary/input provenance, command and resources.

Historical pilot logs/provenance were archived under `attempts/` by the runner. The adjacent summary JSON preserves the final run records and group statistics; detailed large piece/edge files remain in the data directory.

Limitations: no parameter optimum claimed; relative ordinary winding accuracy is not known. Existing legacy reference-pruning/gauge-calibration workflows have not been re-benchmarked with all components retained; use the explicit legacy filter only to reproduce their old cohort. New diagnostic runs use the audited canonical reference gauge instead.
