# Rejections by initial relation type

Moving-window16, raw continuous winding extraction; full Paris4 AFV.
Percentages below use each initial certified relation population as denominator.
Seed failures cannot be classified and are reported separately. Each candidate
retains its INITIAL certified type even if subsequent refinement fails orientation.
All counts include initial and adaptive attempts, not distinct relationships.

## parallel

Denominator: 13,138,261 attempts (40.1816% of all attempts).

| Outcome | Count | % within type |
|---|---:|---:|
| Tangent support after refinement | 360,868 | 2.7467% |
| Orientation after refinement | 131,624 | 1.0018% |
| Central connector >128 voxels | 161 | 0.0012% |
| Probe endpoint support | 2,099,027 | 15.9764% |
| Probe tangent support | 28,755 | 0.2189% |
| Probe orientation | 760,865 | 5.7912% |
| Probe connector >128 voxels | 503,159 | 3.8297% |
| Reciprocal | 0 | 0.0000% |
| Winding variation (disabled) | 0 | 0.0000% |
| Winding category (disabled) | 0 | 0.0000% |
| Winding cutoff (disabled) | 0 | 0.0000% |
| Total rejected | 3,884,459 | 29.5660% |
| deduplicated | 3,206,077 | 24.4026% |
| accepted | 6,047,725 | 46.0314% |

## perpendicular

Denominator: 12,330,351 attempts (37.7107% of all attempts).

| Outcome | Count | % within type |
|---|---:|---:|
| Tangent support after refinement | 769,916 | 6.2441% |
| Orientation after refinement | 164,167 | 1.3314% |
| Central connector >128 voxels | 0 | 0.0000% |
| Probe endpoint support | 281,824 | 2.2856% |
| Probe tangent support | 0 | 0.0000% |
| Probe orientation | 58,718 | 0.4762% |
| Probe connector >128 voxels | 5,512 | 0.0447% |
| Reciprocal | 0 | 0.0000% |
| Winding variation (disabled) | 0 | 0.0000% |
| Winding category (disabled) | 0 | 0.0000% |
| Winding cutoff (disabled) | 0 | 0.0000% |
| Total rejected | 1,280,137 | 10.3820% |
| deduplicated | 9,729,604 | 78.9078% |
| accepted | 1,320,610 | 10.7102% |

## Unclassified seed failures

Denominator here: all 32,697,203 attempts, unlike tables above.

| Outcome | Count | % of all attempts |
|---|---:|---:|
| seed_orientation | 861,553 | 2.6349% |
| seed_tangent_support | 6,367,038 | 19.4727% |

First-failure attribution, not independent filter ablations. Deduplication is
consolidation, not rejection. Probe distance loss is overwhelmingly parallel.
No extraction decisions were changed; all7,368,335serialized measurements match
raw_winding_v1/full.afc.sqlite EXACTLY.141native tests pass. Both per-type and
combined attempt/accepted/dedup/failure ledgers validate.

## Reproduce

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_relation_audit.json
/home/hendrik/.venv_las/bin/python global-fiber-sgd/summarize_rejection_audit.py global-fiber-sgd/data/relation_audit_v1/full.json --output global-fiber-sgd/relation_rejection_statistics.json
```

RelWithDebInfo/GCC13,128threads, one run;359.14s runner wall including full
saved-measurement comparison. No database export/overwrite. Raw report/logs and
run metadata: data/relation_audit_v1. These are diagnostic counts, not speedup
measurements or proof of anatomical correctness.
