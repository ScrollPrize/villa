# Isotropic20 orthogonal-only minimum-length sweep

Same saved raw40500/full-prefix512/isotropic20/no-history traces. Only segmentation changes: no parallel-transition cuts; greedy ascending orthogonal-event midpoint candidates must leave both previous piece and final tail at least the configured minimum. Short whole source traces remain intact. Multiple parallel/orthogonal measurements may share a piece. No changes to certification/refinement/evidence weights, defect/break costs, reference policy or500message budget. 128CPU threads,RelWithDebInfo,one repetition; no inference or retracing.

| Construction | Replay | Constraint | Oracle accuracy | Non-oracle accuracy | Observed shift |
|---|---:|---:|---:|---:|---:|
| overlapping512/128 | 100.00% | -25.24% | 80.77% | 50.00% | 83.33% |
| previous event mode | 100.00% | — | — | 11.54% | 4.17% |
| min8 | 100.00% | — | — | 15.38% | 20.83% |
| min16 | 100.00% | — | — | 15.38% | 16.67% |
| min32 | 100.00% | — | — | 23.08% | 20.83% |
| min64 | 100.00% | — | — | 15.38% | 12.50% |
| min128 | 100.00% | — | — | 15.38% | 20.83% |
| min256 | 100.00% | — | — | 34.62% | 16.67% |

| Minimum | Pieces | Median length | Minimum length | Short intact sources | Constraints | Active components | Missing/tied/unbounded | Oracle / independent seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 18254 | 32.000 | 0.561 | 48 | 254839 | 21 | 2/2/2 | 80.03 / 76.77 |
| 16 | 15882 | 37.581 | 0.561 | 82 | 252467 | 26 | 2/1/1 | 65.01 / 71.05 |
| 32 | 12762 | 48.694 | 0.561 | 164 | 249347 | 18 | 2/1/1 | 41.83 / 44.77 |
| 64 | 8273 | 82.947 | 0.561 | 246 | 244858 | 23 | 2/1/1 | 31.74 / 36.41 |
| 128 | 4842 | 149.041 | 0.561 | 466 | 241427 | 27 | 2/2/2 | 23.75 / 27.86 |
| 256 | 2730 | 281.037 | 0.561 | 728 | 239315 | 29 | 2/4/3 | 19.35 / 23.26 |

Replay inherited unchanged:0ordinary errors,6filtered stops,93.7463%reference distance retained. Accuracy denominator26; observed shift omits missing references and bridges their gaps. Rank1scores are not uniquely determined when tied/unbounded readouts exist. Constraint score−100×problematic/retainedfulfilled (higher better),not removal percent. Oracle calibration failures remain unavailable; no arbitrary offsets or constraint removal. All initial solves retain500messages; convergence details and full ladders in JSON.

## min8

Reference-connected BP gauge has no calibration
local constraint events: pieces=18254 constraints=254839 continuity=16375 consistency_rejected=1531057 support_rejected=0 certified=238464 retained_certified=238464 refined=96283 midpoint_cuts=16375 transition_cuts=0 parallel_omitted=0
reference cross source events: certified=4754 retained=4754 missing_owner=0
message_limit  18254  18254  44720  389  0.002563  0.016181  0.250000  1  500  39.712450  128  578166  -5  5  0  10  0.927204  29.188263

## min16

Reference-connected BP gauge has no calibration
local constraint events: pieces=15882 constraints=252467 continuity=14003 consistency_rejected=1531057 support_rejected=0 certified=238464 retained_certified=238464 refined=96283 midpoint_cuts=14003 transition_cuts=0 parallel_omitted=0
reference cross source events: certified=4754 retained=4754 missing_owner=0
message_limit  15882  15882  41674  389  0.002712  0.041620  0.250000  1  500  46.656523  128  533642  -5  4  0  9  0.932120  31.957968

## min32

Reference-connected BP gauge has no calibration
local constraint events: pieces=12762 constraints=249347 continuity=10883 consistency_rejected=1531057 support_rejected=0 certified=238464 retained_certified=238464 refined=96283 midpoint_cuts=10883 transition_cuts=0 parallel_omitted=0
reference cross source events: certified=4754 retained=4754 missing_owner=0
message_limit  12762  12762  38127  389  0.002986  0.011967  0.250000  1  500  95.927359  128  483704  -6  4  0  10  0.951397  25.140114

## min64

Reference-connected BP gauge has no calibration
local constraint events: pieces=8273 constraints=244858 continuity=6394 consistency_rejected=1531057 support_rejected=0 certified=238464 retained_certified=238464 refined=96283 midpoint_cuts=6394 transition_cuts=0 parallel_omitted=0
reference cross source events: certified=4754 retained=4754 missing_owner=0
message_limit  8273  8273  30395  389  0.003456  0.009733  0.250000  1  500  15.975658  128  376778  -5  4  0  9  0.935077  19.298656

## min128

Reference-connected BP gauge has no calibration
local constraint events: pieces=4842 constraints=241427 continuity=2963 consistency_rejected=1531057 support_rejected=0 certified=238464 retained_certified=238464 refined=96283 midpoint_cuts=2963 transition_cuts=0 parallel_omitted=0
reference cross source events: certified=4754 retained=4754 missing_owner=0
message_limit  4842  4842  24423  389  0.003994  0.005973  0.250000  1  500  143.371160  128  294335  -5  4  0  9  0.903371  12.547371

## min256

Reference-connected BP gauge has no calibration
local constraint events: pieces=2730 constraints=239315 continuity=851 consistency_rejected=1531057 support_rejected=0 certified=238464 retained_certified=238464 refined=96283 midpoint_cuts=851 transition_cuts=0 parallel_omitted=0
reference cross source events: certified=4754 retained=4754 missing_owner=0
message_limit  2730  2730  19562  389  0.004468  0.003407  0.250000  2  500  342.129130  128  244167  -6  7  0  13  0.862645  9.544993

Artifacts:`/ephemeral/hendrik/fiber_crop_bench/20261002_iso20_orthogonal_minimum_v1`. Approved command:`bash /home/hendrik/villa3/scripts/run_training_failure_mining`, control action `iso20_minimum_sweep`. Native build/tests:action `build`.
