# Fractional measurement experiment

Baseline: commit `f281831e8`, saved `global-fiber-min-flow/data/seven_v1`.
New accepted baseline: `global-fiber-min-flow/data/seven_fractional_v1`; default `target_mode: fractional`.
Historical rounded artifacts and source data are unchanged. Rounded mode remains available explicitly through `target_mode: rounded`. The canonical `run_seven.json` and default GT audit now select the fractional baseline.

## Formulation

Keep integer winding labels. For signed measurements, use the continuous center underlying the baseline's integer target selection:

```
outward = scaled_distance - sign * (parallel ? 0 : perpendicular_offset * family_delta)
margin = !parallel && sign * family_delta < 0
measured = sign * max(outward, margin)
t = measured + connector_transport - fiber_b_transport + fiber_a_transport
```

Thus the existing H/V calibration and order clamp remain; this is the clamped calibrated measurement, not raw distance. With `n=floor(t)` and `alpha=t-n`, emit two edges with targets `n`, `n+1` and weights `w*(1-alpha)`, `w*alpha`. Omit zero-weight arcs. For integer predicted differences this equals `w*abs(difference-t)` exactly. Capacities may be fractional; labels and arc costs remain integer. The saved `flow_event_index` maps expanded arcs to original observations.

Unsigned observations do not determine a unique fractional center. Preserve the baseline policy: 814 unique-zero unsigned observations per large case retain target zero; the same 20 ambiguous observations are excluded. Sign-only tests keep their integer hinge objective. All original fibers remain.

## GT results

Same frozen reference and scoring as baseline, with per-component integer gauge alignment. GT source 7 is excluded: small cases score 41/46 relations; large cases score 4918/5251 reference events. Joint includes GT fibers as ordinary input; this is not independent held-out validation.

| Case | Exact GT before | Exact GT fractional |
|---|---:|---:|
| full | 41/41 | 41/41 |
| half | 41/41 | 41/41 |
| double | 41/41 | 41/41 |
| sign | 41/41 | 41/41 |
| sign_sparse_control | 41/41 | 41/41 |
| free | 4670/4918 | 4714/4918 |
| joint | 4754/4918 | 4753/4918 |

| Large case / family | Exact before → fractional | GT sign violations before → fractional |
|---|---:|---:|
| free perpendicular | 491 → 494 / 575 | 4 → 4 / 575 |
| free parallel | 4179 → 4220 / 4343 | 9 → 0 / 4343 |
| joint perpendicular | 495 → 497 / 575 | 3 → 3 / 575 |
| joint parallel | 4259 → 4256 / 4343 | 0 → 0 / 4343 |

Free mean absolute GT error improves 0.050834 → 0.041684; joint remains 0.033550. All nine fixed GT sign violations concern ordinary fiber 423963 versus GT1. There are no new GT violations. Remaining free violations: one reversal on fiber291441 versus GT2, plus missing-next errors on fibers424632,424646,425073 versus GT1. Joint retains the latter three only. Each remaining violation is one winding step.

Disconnected component offsets remain unidentifiable. Under a single global gauge, free GT signs are 5 perpendicular and 27 parallel violations (baseline 5 and 36); joint is unchanged from the component-aligned counts. Headline results use the same per-component policy as before.

94/2053 free and 92/2060 joint node labels change after aligning each component by its modal label difference; maximum change is three steps. In full/half/double only excluded GT fiber7 changes. The sign-only solutions are identical.

Internal measurement-sign violations are a different metric: free22→3, joint24→6, all perpendicular; parallel remains zero.

## Objective and certificate checks

Compare assignments under the same objective; raw costs between formulations are not directly comparable.

| Case | Rounded objective, old → new assignment | Fractional objective, old → new assignment |
|---|---:|---:|
| full/half/double | 15 → 17.5 | 30.133965 → 28.751157 |
| sign/sign_sparse_control | 0 → 0 | 0 → 0 |
| free | 886 → 909 | 3270.844651 → 3232.192059 |
| joint | 1099 → 1127.5 | 4149.353655 → 4108.967315 |

The change therefore does more than select another optimum of the old objective. Both old and new assignments were also checked against direct fractional absolute residuals, independently of expanded-edge cost. All certificates passed: maximum absolute primal/dual gap 2.73e-12, conservation error 7.30e-13, zero capacity violation and zero potential-rounding deviation.

Free flow arcs increase 15869→30179, joint21166→40682. Trial solver times were 0.193s and 0.224s (historical baseline0.130s and0.159s). These are single runs, not a controlled performance benchmark. Full trial processing took7.62s and10.37s respectively.

## Implementation and reproduction

- `fiber_min_flow.py`: calibrated fractional centers and adjacent-target expansion.
- `fiber_min_flow_benchmark.py`: fractional default with explicit rounded option, unchanged scoring, event/arc mapping and manifest metadata.
- `global-fiber-min-flow/compare_fractional.py`: equal input hashes/coverage/components/seam arrays, both objectives, changed labels and summary counts.
- `global-fiber-min-flow/audit_gt_signs.py`: reusable source/output arguments; detailed CSV/JSON results in `data/gt_sign_audit_fractional_v1`.

From this worktree:

```bash
/home/hendrik/.venv_las/bin/python vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config global-fiber-min-flow/run_seven_fractional.json
/home/hendrik/.venv_las/bin/python global-fiber-min-flow/compare_fractional.py
/home/hendrik/.venv_las/bin/python global-fiber-min-flow/audit_gt_signs.py --source global-fiber-min-flow/data/seven_fractional_v1 --output global-fiber-min-flow/data/gt_sign_audit_fractional_v1
PYTHONPATH=vesuvius/src /home/hendrik/.venv_las/bin/python -m pytest -q --noconftest -o pythonpath=vesuvius/src global-fiber-min-flow/test_audit_signs.py vesuvius/tests/test_fiber_min_flow.py vesuvius/tests/test_fiber_uv_bucket_split.py vesuvius/tests/test_fiber_uv_layout.py
```

37 tests passed, including integer-domain equivalence, brute-force fractional optimization, reversal, integer seam/gauge shifts, H/V clamps, unsigned policy and unchanged sign-only behavior. No installation required. Original AFC database remains unavailable as documented for baseline; all consumed data hashes match. No continuous surface construction was changed or evaluated.
