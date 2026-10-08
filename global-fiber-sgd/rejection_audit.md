# Full-volume per-check rejection audit — 2026-10-08

Denominator:31,940,445candidate attempts =30,412,865initial +1,527,580refinement.
Attempts are sampled-point encounters, not unique physical relationships. They
can repeatedly target the same fiber pair/encounter. No thresholds were changed.
All2,334,188accepted full measurement JSON records match the saved DB exactly.

## Mutually exclusive first-failure attribution

| Check | Count | % of all attempts |
|---|---:|---:|
| Initial tangent missing/degenerate (usually endpoint support) | 6,367,038 | 19.9341% |
| Initial orientation ambiguous/inconsistent | 835,249 | 2.6150% |
| Tangent support after closest-point refinement | 1,044,350 | 3.2697% |
| Orientation after refinement | 288,808 | 0.9042% |
| Central reciprocal projection exceeds8vx | 8,159,623 | 25.5464% |
| Insufficient endpoint support for probes | 1,043,266 | 3.2663% |
| Probe tangent missing/degenerate | 3,358 | 0.0105% |
| Probe orientation inconsistent | 257,769 | 0.8070% |
| Probe reciprocal projection exceeds8vx | 1,973,367 | 6.1783% |
| Probe connector exceeds128vx | 17,589 | 0.0551% |
| Winding difference exceeds0.25 | 199,269 | 0.6239% |
| Rounded winding category differs | 874,065 | 2.7365% |
| Winding distance >=1.5 | 958,707 | 3.0015% |
| Duplicate encounter removed | 7,583,799 | 23.7436% |
| Accepted | 2,334,188 | 7.3079% |

No first failures attributed to central distance, monotonic correspondence,
piece bounds or invalid winding samples. Sign checks were not applied (unsigned
extraction). The ledger sums exactly to all attempts. Consistency alone is65.9470%.

Reciprocal checks combined31.7246%; tangent/endpoint support combined26.4806%;
orientation combined4.3263%. Missing/degenerate tangents share a category, so
do not claim all26.48% are specifically near endpoints.

These are **first failures in execution order**, not standalone filter ablations.
A candidate failing early may fail other tests too. Later geometric checks need
a valid encounter/type and cannot be interpreted on every earlier failure.
Thus disabling a check does not imply recovering its entire count as constraints.
The initial128vx discovery cutoff is outside this denominator: points never
discovered cannot appear as rejected attempts. Probe-distance counts do not
measure loss caused by the initial search radius.

## Independently evaluated winding predicates (overlapping)

5,193,486attempts reached winding certification, including candidates later
deduplicated. Each predicate counts a candidate once even if several probes fail.

| Predicate | Count | % all attempts | % reaching winding certification |
|---|---:|---:|---:|
| Difference exceeds0.25 | 239,700 | 0.7505% | 4.6154% |
| Rounded category differs | 944,847 | 2.9582% | 18.1929% |

These overlap and must not be summed. All sampled windings here were valid;
missing/out-of-range values in other workloads are not diagnostic passes.

## Reproduce / validation

Approved build and runner:

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_rejection_audit.json
/home/hendrik/.venv_las/bin/python global-fiber-sgd/summarize_rejection_audit.py global-fiber-sgd/data/rejection_audit_v1/full.json --output global-fiber-sgd/rejection_audit_results.json
```

Native137tests passed before fullrun; slab19,889/full2,334,188measurement
comparisons are exact serialized-field comparisons, not only aggregate counts.
Full diagnostic process251.63s including loading and comparison; extraction196.22s.
Original constraint database opened read-only, never overwritten.
Detailed counters in rejection_audit_results.json; raw configs/results/run hashes
under data/rejection_audit_v1. Added afterward: defensive diagnostic range guard
for extreme custom sampler values; final guard validation uses separate runner
run_audit_guard_validation.json and does not repeat the fullvolume run.
Final guard validation:138native tests passed; all19,889pilot measurements still
match the saved database exactly.
