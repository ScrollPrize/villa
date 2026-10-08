# Reference-seeded reverse diagnostic

Source: reviewed_fibers_20260930_v3/reverse256_v1. Only remaining counted ordinary failures after periodic filtering are probed. Historical Gaussian12/lowres15 included for comparison. This diagnostic is not a new full-cohort score.

Native beam8/lookahead2; unchanged model-specific step4 prediction voxels (full-resolution models8base, lowres32base). All probes run through shared live_evaluate.run/native parallel evaluation; same inference namespaces and checkpoints, demand-managed GPUs0–5. No production filter changes.

Each failure has a reverse control along the final256base voxels of its saved erroneous trace, plus GT-tangent-seeded reverse probes at failure arc and +128/+256base along the original direction. GT probes end at max(segment start,failure−256). Offsets are clipped to the eligible span. These are directed cases, not necessarily distinct fibers.

Online lateral stopping uses a finite1e9 radius; no periodic filtering, zero grace. Native endpoint/progress and unchanged step budget still bound each run. Score only the first uninterrupted segment. “Correct” requires reference_end termination AND every saved anisotropic absolute error ≤20 (tight20/40ellipse) or ≤40 (regular40/80ellipse). Merely reaching the projected endpoint is not sufficient. The control uses ordinary reference matching, not precisely the production local reverse checker.

Convergence is conservative: ≥80base contiguous travel within10Euclidean base voxels and25° unsigned tangent agreement of saved wrong-branch segments; both paths must be outside regularGT40/80. The seed has no native match: diagnostic seed error is exactly zero by construction; the original segment first edge is conservatively excluded. No detection is not proof of no convergence.

Run: set scripts/training_failure_mining_control.json to action=reviewed_reference_reverse and reviewed_model=gaussian or lowres, then `bash /home/hendrik/villa3/scripts/run_training_failure_mining`. analysis_only=true summarizes available artifacts without running probes.

For control rows, “within reference” means reproducing the saved **wrong route**, NOT correctness against GT. For GT rows it means following GT.

| Model | Probe | Completed artifacts | Within tight supplied reference | Within regular supplied reference |
|---|---|---:|---:|---:|
| gaussian | control | 12/12 | 8/12 (66.7%) | 10/12 (83.3%) |
| gaussian | gt0 | 12/12 | 8/12 (66.7%) | 9/12 (75.0%) |
| gaussian | gt128 | 12/12 | 8/12 (66.7%) | 9/12 (75.0%) |
| gaussian | gt256 | 12/12 | 8/12 (66.7%) | 10/12 (83.3%) |
| lowres | control | 15/15 | 13/15 (86.7%) | 13/15 (86.7%) |
| lowres | gt0 | 15/15 | 12/15 (80.0%) | 13/15 (86.7%) |
| lowres | gt128 | 15/15 | 10/15 (66.7%) | 12/15 (80.0%) |
| lowres | gt256 | 15/15 | 11/15 (73.3%) | 13/15 (86.7%) |
| halfsigma_sideband_best_raw_20261002 | control | 13/13 | 7/13 (53.8%) | 10/13 (76.9%) |
| halfsigma_sideband_best_raw_20261002 | gt0 | 13/13 | 11/13 (84.6%) | 13/13 (100.0%) |
| halfsigma_sideband_best_raw_20261002 | gt128 | 13/13 | 9/13 (69.2%) | 10/13 (76.9%) |
| halfsigma_sideband_best_raw_20261002 | gt256 | 13/13 | 9/13 (69.2%) | 10/13 (76.9%) |

## gaussian

Artifacts: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/reference_reverse_diagnostic_v1/gaussian`. Results: `results.json`. SHA256 `274acb0c686bfb030646d039f2740f7ad6349cd507a4de0843cf26e4f638ccec`.

All requested probes complete.

Cells below are max anisotropic error in base voxels. † means the first segment did not reach the reference endpoint. — means pending. Event indices are zero-based.

| Scroll | Original case/event | Control | GT0 | GT+128 | GT+256 |
|---|---:|---:|---:|---:|---:|
| PHerc0139 | 4/0 | 5.2 | 4.9 | 4.4 | 6.3 |
| PHerc0139 | 5/0 | 4.6 | 9.1 | 7.4 | 8.1 |
| PHerc0139 | 6/0 | 78.5 | 142.3 | 511.7† | 640.1† |
| PHerc0139 | 7/0 | 5.0 | 110.3 | 686.7† | 914.3† |
| PHerc1299 | 57/0 | 2.1 | 4.7 | 7.7 | 8.8 |
| PHerc1299 | 60/0 | 3.9 | 5.7 | 5.5 | 4.1 |
| PHerc1299 | 61/0 | 22.2 | 19.9 | 19.9 | 19.9 |
| PHerc1299 | 62/0 | 4.2 | 4.1 | 3.9 | 3.3 |
| PHerc1299 | 66/0 | 1.5 | 49.9 | 49.1 | 8.4 |
| PHercParis4 | 104/0 | 162.9 | 23.0 | 30.8 | 27.8 |
| PHercParis4 | 117/0 | 21.7 | 4.5 | 16.9 | 20.4 |
| PHercParis4 | 170/0 | 3.3 | 5.7 | 4.5 | 5.3 |

Complete four-probe groups: 12; wrong-route control tight AND at least one GT probe tight: 7. Sustained wrong-branch convergence: 0/33 testable GT probes.

Tight diagnostic partition: 4 fail the saved-wrong-route reverse control; 7 reproduce the wrong route but also have a correct GT-seeded alternative; 1 reproduce the wrong route with no tight-correct GT probe. The last count is an observed diagnostic remainder, not a proven count of undetectable errors.

Interpretation: independently reversible wrong and correct paths can coexist. This supports investigating competing-path evidence, but does not establish a usable ground-truth-free filter. No precision/recall claim is possible without controls on successful traces.

Clipped offsets (base voxels; duplicate seeds are not independent evidence): 61_0_gt128: requested+128, effective+0.00; 61_0_gt256: requested+256, effective+0.00; 66_0_gt256: requested+256, effective+188.41.

## lowres

Artifacts: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/reference_reverse_diagnostic_v1/lowres`. Results: `results.json`. SHA256 `67edd3a1bde162c7105ee37d849a0f0b60a9b840dd37fc7ff6e17e0c3a2e9742`.

All requested probes complete.

Cells below are max anisotropic error in base voxels. † means the first segment did not reach the reference endpoint. — means pending. Event indices are zero-based.

| Scroll | Original case/event | Control | GT0 | GT+128 | GT+256 |
|---|---:|---:|---:|---:|---:|
| PHerc0139 | 4/0 | 4.2 | 4.5 | 5.1 | 5.8 |
| PHerc0139 | 5/0 | 4.2 | 22.5 | 24.1 | 24.4 |
| PHerc0139 | 6/0 | 73.0 | 367.5† | 527.1† | 653.8† |
| PHerc0139 | 7/0 | 4.6 | 224.4† | 713.3† | 946.5† |
| PHerc0139 | 23/0 | 8.2 | 6.2 | 5.5 | 5.7 |
| PHerc1299 | 60/0 | 1.6 | 8.2 | 9.2 | 9.3 |
| PHerc1299 | 62/0 | 1.6 | 5.1 | 5.5 | 5.1 |
| PHerc1299 | 66/0 | 5.3 | 17.5 | 17.9 | 17.9 |
| PHerc1299 | 67/0 | 2.6 | 5.8 | 23.3 | 25.5 |
| PHercParis4 | 74/0 | 4.8 | 4.5 | 5.7 | 5.2 |
| PHercParis4 | 94/0 | 3.7 | 6.4 | 7.2 | 6.1 |
| PHercParis4 | 129/0 | 362.8† | 7.2 | 54.8 | 15.9 |
| PHercParis4 | 129/1 | 4.4 | 8.3 | 7.6 | 18.3 |
| PHercParis4 | 137/0 | 4.0 | 7.1 | 6.3 | 5.5 |
| PHercParis4 | 170/0 | 4.8 | 6.8 | 11.8 | 10.4 |

Complete four-probe groups: 15; wrong-route control tight AND at least one GT probe tight: 11. Sustained wrong-branch convergence: 0/45 testable GT probes.

Tight diagnostic partition: 2 fail the saved-wrong-route reverse control; 11 reproduce the wrong route but also have a correct GT-seeded alternative; 2 reproduce the wrong route with no tight-correct GT probe. The last count is an observed diagnostic remainder, not a proven count of undetectable errors.

Interpretation: independently reversible wrong and correct paths can coexist. This supports investigating competing-path evidence, but does not establish a usable ground-truth-free filter. No precision/recall claim is possible without controls on successful traces.

Clipped offsets (base voxels; duplicate seeds are not independent evidence): 66_0_gt128: requested+128, effective+102.80; 66_0_gt256: requested+256, effective+102.80.

## halfsigma_sideband_best_raw_20261002

Artifacts: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/reference_reverse_diagnostic_v1/halfsigma_sideband_best_raw_20261002`. Results: `results.json`. SHA256 `9d3bb1c0cd5b0ee58d16d9b2bdcec931faec675421e6e2c707f7fd10ce059f7e`.

All requested probes complete.

Cells below are max anisotropic error in base voxels. † means the first segment did not reach the reference endpoint. — means pending. Event indices are zero-based.

| Scroll | Original case/event | Control | GT0 | GT+128 | GT+256 |
|---|---:|---:|---:|---:|---:|
| PHerc0139 | 6/0 | 77.6 | 35.9 | 512.6† | 637.1† |
| PHerc0139 | 7/0 | 4.8 | 5.8 | 685.6† | 911.9† |
| PHerc0139 | 8/0 | 1.9 | 4.9 | 5.5 | 3.7 |
| PHerc0139 | 33/0 | 16.2 | 5.4 | 5.6 | 4.7 |
| PHerc0139 | 40/0 | 39.4 | 5.0 | 4.5 | 3.8 |
| PHerc0139 | 41/0 | 25.8 | 4.0 | 3.4 | 5.3 |
| PHerc1299 | 60/0 | 5.0 | 4.5 | 5.6 | 5.6 |
| PHerc1299 | 61/0 | 31.9 | 19.7 | 19.7 | 19.7 |
| PHerc1299 | 66/0 | 2.2 | 8.0 | 8.0 | 8.0 |
| PHercParis4 | 128/0 | 435.6† | 27.4 | 448.0† | 159.1 |
| PHercParis4 | 136/0 | 45.6 | 4.7 | 21.5 | 22.3 |
| PHercParis4 | 143/0 | 4.0 | 4.5 | 6.0 | 4.5 |
| PHercParis4 | 166/0 | 3.7 | 3.0 | 5.1 | 5.4 |

Complete four-probe groups: 13; wrong-route control tight AND at least one GT probe tight: 7. Sustained wrong-branch convergence: 0/36 testable GT probes.

Tight diagnostic partition: 6 fail the saved-wrong-route reverse control; 7 reproduce the wrong route but also have a correct GT-seeded alternative; 0 reproduce the wrong route with no tight-correct GT probe. The last count is an observed diagnostic remainder, not a proven count of undetectable errors.

Interpretation: independently reversible wrong and correct paths can coexist. This supports investigating competing-path evidence, but does not establish a usable ground-truth-free filter. No precision/recall claim is possible without controls on successful traces.

Clipped offsets (base voxels; duplicate seeds are not independent evidence): 61_0_gt128: requested+128, effective+0.00; 61_0_gt256: requested+256, effective+0.00; 66_0_gt256: requested+256, effective+185.58.

## Full-cohort periodic-filter comparison

Same 91 fibers / 182 directions. Filtering uses 256-base-voxel windows and a 20/40 ellipse; GT-assisted reseeding uses 10/20 with a local presence peak and direction alignment. Filtered stops are not ordinary restarts. Skipped regions receive no retained-length credit.

| Model / mode | Ordinary restarts | Filtered stops | Retained % | Skipped mm |
|---|---:|---:|---:|---:|
| gaussian/plain | 32 | 0 | 99.999 | 0.000 |
| gaussian/filtered | 12 | 62 | 97.293 | 37.508 |
| lowres/plain | 29 | 0 | 100.000 | 0.000 |
| lowres/filtered | 15 | 51 | 97.702 | 31.850 |
| halfsigma_sideband_best_raw_20261002/plain | 16 | 0 | 100.000 | 0.000 |
| halfsigma_sideband_best_raw_20261002/filtered | 13 | 36 | 98.438 | 21.650 |

### Current model per scroll

| Scroll | Plain ordinary | Filtered ordinary | Filtered stops | Retained % |
|---|---:|---:|---:|---:|
| PHerc0139 | 8 | 6 | 12 | 97.503 |
| PHerc0332 | 0 | 0 | 0 | 100.000 |
| PHerc1299 | 2 | 3 | 6 | 97.392 |
| PHerc1451 | 0 | 0 | 0 | 100.000 |
| PHercParis4 | 6 | 4 | 18 | 98.854 |
