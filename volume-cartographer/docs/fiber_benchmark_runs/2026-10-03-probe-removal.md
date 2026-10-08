# Reference-conditioned constraint removal: probe comparison

Selected baseline (user decision): **median +/-64, minimum256, parallel magnitude
weight0.05, disabling cost150 per128 base voxels, gap boundary0**. Current best
fine-split candidate; the historical overlapping baseline still has a better
removal score, so this is not a claim of global dominance. Production defaults
unchanged. Follow-up cost sweep records its own results separately.

Existing oracle-inliers benchmark, one oracle round, magnitude weight1, accept-message-limit as historical baseline. Frozen raw40500 iso20 1024 crop; fine256, soft parallel weight0.05, hard signs both, old reference extraction. 128CPU threads,500-message limit, RelWithDebInfo. No inference/retracing or geometric matching audit.

| Variant | Initial constraints | Problematic | Problematic / initial | Retained fulfilled | Constraint score ↑ | Oracle accuracy | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| winner-repeat | 243682 | 56027 | 22.9919% | 187655 | -29.8564% | 76.9231% | 239.6 |
| winner-median64 | 243682 | 55515 | 22.7817% | 188167 | -29.5030% | 76.9231% | 238.4 |
| winner-consensus32 | 243682 | 55816 | 22.9053% | 187866 | -29.7105% | 76.9231% | 238.7 |

Constraint score = −100×problematic/retained-fulfilled (higher/closer to zero is better). Problematic includes constraints incident to removed pieces, retained infringed constraints and retained defect constraints. It is the established operational removal score, not proof each removed measurement is individually geometrically false.

| Variant | Removed-incident | Retained-infringed | Retained-defect |
|---|---:|---:|---:|
| winner-repeat | 32326 | 23701 | 0 |
| winner-median64 | 32137 | 23378 | 0 |
| winner-consensus32 | 32280 | 23536 | 0 |

Artifacts/configs/provenance/logs: `/ephemeral/hendrik/fiber_crop_bench/20261003_iso20_probe_removal_v1`. Independent scores for all three:61.54% exact,87.50% observed shift (previous runs). Historical overlap result:constraint−25.2365%,oracle80.7692%; different subdivision/objective, not a probe-only control.

Run approved `bash /home/hendrik/villa3/scripts/run_training_failure_mining` with action `iso20_probe_removal`. Shared native benchmark runner and parser reused. Probe semantics unchanged: four displaced connectors plus center median; consensus disagreement retains original, not rejection. This is not an endpoint-agreement experiment.

## Paired comparison

All variants contain243682 initial constraints; no changed acceptance or segmentation.
- winner-median64: -512 problematic constraints versus no probes.
- winner-consensus32: -211 problematic constraints versus no probes.

Initial states and original measurement exports were compared to each saved independent run; see initial_states_identical and initial_factors_identical in JSON. Eight Python checks passed; no native code/build changes. One crop, one run per variant; these are the established heuristic removal outcomes, not a globally minimal deletion proof.
