# Anchor placement coverage

Canonical anchors, nearest GT projection, configured GT acceptance ellipse and flat endpoint caps.
Gaps are GT arclength / anchor-cell side; not literal crossed grid cells.
Interior and endpoint gaps are separate; empty pieces remain in occupancy statistics.

Cell sides: [32.0] base voxels. Runtime: 52.46s.

| Scroll | Pieces | Accepted anchors | Empty pieces | Median gap (cells) | P95 | Max | Empty arc bins |
|---|---:|---:|---:|---:|---:|---:|---:|
| All | 200 | 69141 | 0 | 0.070 | 0.445 | 3.736 | 0.68% |
| PHerc0139 | 77 | 23620 | 0 | 0.068 | 0.413 | 1.219 | 0.54% |
| PHerc0332 | 1 | 139 | 0 | 0.108 | 0.866 | 1.225 | 0.00% |
| PHerc1299 | 25 | 5129 | 0 | 0.090 | 0.818 | 1.254 | 1.48% |
| PHerc1451 | 1 | 97 | 0 | 0.139 | 0.964 | 1.049 | 6.90% |
| PHercParis4 | 96 | 40156 | 0 | 0.069 | 0.408 | 3.736 | 0.56% |

## Pooled interior-gap histogram

| Gap (cell-side units) | Count | Percent |
|---|---:|---:|
| [0, 1) | 67758 | 99.847% |
| [1, 2) | 101 | 0.149% |
| [2, 3) | 0 | 0.000% |
| [3, 4) | 3 | 0.004% |
| [4, 5) | 0 | 0.000% |
| [5, 6) | 0 | 0.000% |
| [6, 7) | 0 | 0.000% |
| [7, 8) | 0 | 0.000% |
| [8, 16) | 0 | 0.000% |
| [16, 32) | 0 | 0.000% |
| [32, ∞) | 0 | 0.000% |

All quantiles pool individual gaps. Arc-bin occupancy is phase-dependent; final partial bins count.
This targeted failure set is not an unbiased sample of the full scrolls.
See metrics.json for endpoint-gap and empty-bin-run distributions, and piece_ranking.json
for worst-gap-first inspection links. Native anchors.json files contain positions and original GT arcs.

## Reproduction and validation

Dataset: locked short_failures_200_20260916_v3,200pieces,662.089848mm. Model: best EMA999u22000, SHA25667edd3a1bde162c7105ee37d849a0f0b60a9b840dd37fc7ff6e17e0c3a2e9742. RelWithDebInfo GCC13 workspace native build; extraction parameters unchanged. One measured run,52.461s including setup/cleanup and demand GPU allocation. Predominantly cached anchors/predictions;40new inference tiles,9publishedchunks in Paris4. No fiberlet construction. Initialv1 failed after77pieces at coordinator restart-summary correction; fixed/tested, retained as incomplete history.

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_short200_anchor_coverage_u22000_20260916_v2.json
```

Use a fresh output_root to repeat.122native cases and261Python tests passed,8optional skips. No numerical extraction changes or new dependencies. CPU matching/anchor construction native; Python handles orchestration/analysis.

Artifacts: [report](/ephemeral/hendrik/fiber3d_train2/evaluations/short200_anchor_coverage_u22000_20260916_v2/report.md), [metrics](/ephemeral/hendrik/fiber3d_train2/evaluations/short200_anchor_coverage_u22000_20260916_v2/metrics.json), [worst-piece ranking](/ephemeral/hendrik/fiber3d_train2/evaluations/short200_anchor_coverage_u22000_20260916_v2/piece_ranking.json).

## Additional observations

69141accepted anchor-to-piece matches;67862interior gaps after collapsing identical projected arcs. All200pieces have multiple accepted positions; no invalid-normal fallback in117954ellipse tests. Empty arc bins59/8721(0.6765%); length-weighted empty-bin fraction0.3652%.45empty-bin runs:38length1,2length2,4length3,1length5. Endpoint gaps:median0.1097cells,P950.5947,max4.7911(153.32basevoxels). Worst endpoint piece Paris4 localcase5, fiber030_f101d328e0211cebaeb5.json. Worst interior piece Paris4 localcase19, fiber080_5ced752d764d73a98440.json:3.7364cells(119.56basevoxels).

These measurements suggest widespread anchor absence is not the dominant issue at the current40/80ellipse; they do not measure direction compatibility, connectivity, path quality or traceability. Multiple neighboring fibers can contribute anchors inside the ellipse. Small0332/1451groups each contain only1piece; not broad scroll statistics. No algorithm optimization selected from this baseline yet.
