# Generated beam traces matched to reference fibers

Superseded for source-fiber equivalence by [reference-local ellipse matching](2026-09-23-source-trace-ellipse-matches.md). The spherical thresholds and BP-piece matching below are historical diagnostics, not the requested whole-source-fiber matching criterion.

1024³ crop, new beam8 traces; diagnostic from existing artifacts, no tracing rerun.

Source directory: `/ephemeral/hendrik/fiber_crop_bench/20260923_v1`; traces `beam-crop-new.obj`; constraint/state prefix `bp-beam-new-perpsign-p1low-zero-fixed`.

## Matching method

485 generated traces, 26 references; 24 references intersect the crop. Winding labels 4 and 12 are outside. Arc samples approximately every 4 base voxels, exact point-to-segment distance, nearest dense tangent within 25 degrees (sign invariant). Equivalent means at least 90% reference coverage AND 90% generated-trace coverage. OBJ coordinate precision and sampling make these geometric diagnostics approximate, not exact replay metrics.

| Distance tolerance (base vx) | References with equivalent | Equivalent traces | Reference length covered by any trace |
|---|---:|---:|---:|
| 8 | 7/24 | 8 | 73.20% |
| 16 | 19/24 | 24 | 94.93% |
| 32 | 22/24 | 30 | 99.03% |

## Per-reference matches at 16 base voxels

Trace indices are zero-based OBJ order. Pieces are BP's subdivided trace pieces; a piece is matched only when >=90% lies near exactly one reference.

| Reference winding | Best trace | Best coverage | Any-trace coverage | Equivalent trace count | Matched pieces / non-defect |
|---|---:|---:|---:|---:|---:|
| 0 | 337 | 95.24% | 95.24% | 1 | 2 / 0 |
| 0.5 | 86 | 100.00% | 100.00% | 2 | 11 / 11 |
| 1 | 14 | 100.00% | 100.00% | 1 | 3 / 3 |
| 1.5 | 24 | 98.66% | 98.66% | 1 | 3 / 3 |
| 2 | 31 | 100.00% | 100.00% | 1 | 3 / 3 |
| 2.5 | 5 | 100.00% | 100.00% | 1 | 4 / 4 |
| 3 | 200 | 88.00% | 88.00% | 0 | 1 / 0 |
| 3.5 | 36 | 97.24% | 97.24% | 1 | 4 / 4 |
| 4 | outside crop | — | — | — | — |
| 4.5 | 70 | 68.79% | 68.79% | 0 | 1 / 1 |
| 5 | 11 | 99.21% | 99.21% | 1 | 5 / 4 |
| 5.5 | 289 | 67.16% | 82.84% | 0 | 1 / 1 |
| 6 | 73 | 85.29% | 85.29% | 0 | 2 / 2 |
| 6.5 | 26 | 100.00% | 100.00% | 1 | 1 / 1 |
| 7 | 62 | 95.60% | 95.97% | 2 | 5 / 5 |
| 7.5 | 209 | 96.35% | 96.35% | 1 | 2 / 0 |
| 8 | 0 | 0.00% | 0.00% | 0 | 0 / 0 |
| 8.5 | 16 | 100.00% | 100.00% | 2 | 9 / 9 |
| 9 | 145 | 100.00% | 100.00% | 1 | 3 / 0 |
| 9.5 | 81 | 100.00% | 100.00% | 1 | 4 / 4 |
| 10 | 115 | 100.00% | 100.00% | 1 | 1 / 1 |
| 10.5 | 9 | 100.00% | 100.00% | 3 | 6 / 6 |
| 11 | 6 | 100.00% | 100.00% | 1 | 1 / 1 |
| 11.5 | 104 | 100.00% | 100.00% | 1 | 1 / 0 |
| 12 | outside crop | — | — | — | — |
| 12.5 | 50 | 100.00% | 100.00% | 1 | 1 / 1 |

## Constraints between matched pieces of consecutive reference windings

74 uniquely matched pieces, no ambiguous pieces. These are saved INITIAL ordinary BP states, not the later reference-conditioned final score. Defect-neutralized means at least one endpoint is a defect. The table counts stored measurements, not distinct fiber pairs. No counted pair is a same-trace continuity edge.

| Reference pair | Perpendicular sign measurements | Wrong signs | Defect-neutralized signs | Enabled parallel-one measurements | Wrong parallel targets | Defect-neutralized parallel |
|---|---:|---:|---:|---:|---:|---:|
| 0-0.5 | 7 | 0 | 7 | 0 | 0 | 0 |
| 0-1 | 0 | 0 | 0 | 4 | 0 | 4 |
| 0.5-1 | 18 | 0 | 0 | 0 | 0 | 0 |
| 0.5-1.5 | 0 | 0 | 0 | 24 | 0 | 0 |
| 1-1.5 | 4 | 0 | 0 | 0 | 0 | 0 |
| 1.5-2 | 4 | 0 | 0 | 0 | 0 | 0 |
| 2-2.5 | 4 | 0 | 0 | 0 | 0 | 0 |
| 2-3 | 0 | 0 | 0 | 3 | 0 | 3 |
| 2.5-3 | 4 | 0 | 4 | 0 | 0 | 0 |
| 2.5-3.5 | 0 | 0 | 0 | 12 | 0 | 0 |
| 3-3.5 | 2 | 0 | 2 | 0 | 0 | 0 |
| 5.5-6.5 | 0 | 0 | 0 | 1 | 0 | 0 |
| 6.5-7.5 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7-7.5 | 2 | 0 | 2 | 0 | 0 | 0 |
| 8.5-9 | 12 | 2 | 12 | 0 | 0 | 0 |
| 8.5-9.5 | 0 | 0 | 0 | 28 | 0 | 0 |
| 9-9.5 | 6 | 0 | 6 | 0 | 0 | 0 |
| 10-10.5 | 3 | 0 | 0 | 0 | 0 | 0 |
| 10-11 | 0 | 0 | 0 | 1 | 0 | 0 |
| 10.5-11 | 3 | 0 | 0 | 0 | 0 | 0 |
| 11-11.5 | 1 | 0 | 1 | 0 | 0 | 0 |
| 11.5-12.5 | 0 | 0 | 0 | 0 | 0 | 0 |

Totals: 70 perpendicular sign measurements, 68 correct and 2 wrong; 34 neutralized by defects, including both wrong ones. 73 enabled parallel-one measurements, all correct, 7 neutralized. Six additional parallel measurements target zero or two and are disabled under the current policy. Pairs absent from the table have no measurements between pieces passing this strict geometric match (not proof of no connections through other predicted pieces).

Notable: 8.5–9 has 12 sign measurements, two reversed; all 12 are neutralized because matched reference-9 pieces are defects. Reference 11.5 has a 100%-matching trace (104), but its sole piece 251 is a defect. Its 11–11.5 order measurement is therefore neutralized; 11.5–12.5 has target zero, disabled by the parallel-one-only policy. Reference 8 has no match even at 32 vx; only ~124 base voxels of it lie inside this crop. Reference 12 lies outside the crop entirely.

Conclusion: good equivalent geometry is largely present, but the corresponding ordering ladder is not kept intact in ordinary BP. This identifies gaps/defects to investigate; it does not by itself prove the cause of each error in the later conditioned reference score.
