# Reference-seeded reverse diagnostic

Source: reviewed_fibers_20260930_v3/reverse256_v1. Only the remaining counted ordinary failures are tested: Gaussian12, lowres15. This is not a new full-cohort score.

Native beam8/lookahead2; unchanged model-specific step4 prediction voxels (Gaussian8base, lowres32base). All probes run through shared live_evaluate.run/native parallel evaluation; same inference namespaces and checkpoints, demand-managed GPUs0–5. No production filter changes.

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

Interpretation: independently reversible wrong and correct paths can coexist. This supports investigating competing-path evidence, but does not establish a usable ground-truth-free filter. No precision/recall claim is possible without controls on successful traces.

Clipped offsets (base voxels; duplicate seeds are not independent evidence): 66_0_gt128: requested+128, effective+102.80; 66_0_gt256: requested+256, effective+102.80.
