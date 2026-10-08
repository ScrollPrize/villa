# Source-trace matching using reference-local ellipses

Supersedes the spherical matching diagnostic in `2026-09-23-beam-reference-matches.md`. This analysis uses complete source traces; NO BP pieces, states, classifications, or constraints participate in selecting matches.

## Inputs and method

- Directory: `/ephemeral/hendrik/fiber_crop_bench/20260923_v1`.
- Generated source traces: `beam-crop-new.obj` (485 whole traces).
- Reference polylines: `bp-beam-new-perpsign-p1low-zero-fixed_reference_hs_0.obj` through `_25.obj` (full references, restricted to the 1024³ crop for matching).
- Crop XYZ: [10240,22016,6144] to [11264,23040,7168).
- Normals: native `vc.volume.NormalSampler`, Lasagna sd2-5b17ff6c manifest; sampled at reference positions. All sampled normals valid.
- Ellipse semi-axes: 20 base voxels normal to sheet, 80 in-sheet. Scale s multiplies both axes; squared normalized distance is dn²/400 + dt²/6400.
- Search minimizes ellipse distance itself across reference/source samples, NOT Euclidean distance followed by reweighting. Tangents must agree within 25°, modulo sign.
- Both complete polylines are arc-sampled at <=2 base voxels. No source trace is trimmed to disguise off-reference sections. Reported minima are sampled approximations, additionally limited by OBJ coordinate rounding (~0.05 base voxels). Not exact continuous containment guarantees.
- Coverage = fraction of reference length matched; purity = fraction of source trace inside reference corridor, both including tangent agreement.
- Smallest95 = minimum ellipse scale across all 485 whole traces achieving >=95% coverage AND >=95% purity. Smallest100 similarly covers all sampled points. Matching is independent of BP and does not split source traces.

## Results

22/24 in-crop references have one whole-trace match at >=95% coverage/purity. 20/24 have a fully contained, fully covering sampled match. Across all references there are 35 qualifying source traces: additional matches are redundant alternatives, not inferred restart pieces. All 22 good references fit the 95% criterion at ellipse scale <=0.580 (semi-axes <=11.6/46.4 base voxels).

| Reference winding | Best95 source trace | Ref coverage at s=1 | Trace purity at s=1 | Smallest95 scale | Smallest100 scale / source | Qualifying whole traces (95%) |
|---|---:|---:|---:|---:|---|---:|
| 0 | 337 | 100.00% | 98.55% | 0.333 | 2.586 / 18 | 1 |
| 0.5 | 86 | 100.00% | 100.00% | 0.231 | 0.277 / 86 | 3 |
| 1 | 14 | 100.00% | 100.00% | 0.190 | 0.203 / 14 | 1 |
| 1.5 | 24 | 98.99% | 100.00% | 0.273 | no angle-compatible match | 2 |
| 2 | 31 | 100.00% | 100.00% | 0.275 | 0.363 / 31 | 1 |
| 2.5 | 5 | 100.00% | 100.00% | 0.213 | 0.364 / 5 | 1 |
| 3 | 200 | 100.00% | 100.00% | 0.281 | 0.347 / 200 | 1 |
| 3.5 | 36 | 100.00% | 100.00% | 0.435 | 0.683 / 36 | 2 |
| 4 | outside crop | — | — | — | — | — |
| 4.5 | 70 | 100.00% | 100.00% | 0.580 | 0.636 / 70 | 2 |
| 5 | 11 | 100.00% | 100.00% | 0.511 | 0.531 / 11 | 1 |
| 5.5 | 289 | 91.44% | 78.22% | 1.980 | 2.220 / 289 | 0 |
| 6 | 73 | 100.00% | 100.00% | 0.469 | 0.661 / 73 | 2 |
| 6.5 | 26 | 100.00% | 100.00% | 0.230 | 0.233 / 26 | 1 |
| 7 | 62 | 100.00% | 100.00% | 0.276 | 0.338 / 62 | 2 |
| 7.5 | 209 | 100.00% | 100.00% | 0.294 | 0.575 / 209 | 1 |
| 8 | 97 | 0.00% | 0.00% | 3.710 | 3.736 / 97 | 0 |
| 8.5 | 273 | 100.00% | 100.00% | 0.240 | 0.301 / 16 | 3 |
| 9 | 145 | 100.00% | 100.00% | 0.274 | 0.305 / 145 | 1 |
| 9.5 | 81 | 100.00% | 100.00% | 0.357 | 0.445 / 81 | 1 |
| 10 | 115 | 100.00% | 100.00% | 0.334 | 0.357 / 115 | 2 |
| 10.5 | 9 | 100.00% | 100.00% | 0.417 | 0.489 / 9 | 4 |
| 11 | 6 | 100.00% | 100.00% | 0.312 | 0.449 / 6 | 1 |
| 11.5 | 104 | 100.00% | 100.00% | 0.163 | 0.173 / 104 | 1 |
| 12 | outside crop | — | — | — | — | — |
| 12.5 | 50 | 100.00% | 100.00% | 0.162 | 0.181 / 50 | 1 |

## Exceptions and interpretation

- Reference 0: trace337 covers 100% of reference, 98.55% trace purity; its remaining samples fail the tangent gate. Reference1.5: trace24 covers98.99%, purity100%; remaining reference samples fail the tangent gate. These are not missing complete geometrical tracks under the >=95% criterion.
- Reference5.5: trace289 covers91.44% but only78.22% of that source trace matches. No >=95%-pure source trace exists for this reference under the acceptance ellipse; joining such pure traces cannot resolve it. Expanding to ~1.98 times the ellipse makes the best single trace pass95%, which is outside the intended tolerance and is NOT called a good match.
- Reference8: only~124 base voxels lie inside the crop; no accepted whole-trace equivalent. Best95 requires3.71 times the ellipse and has zero coverage/purity at s=1.
- References4 and12 have no in-crop samples and are excluded from the24-reference matching denominator.
- Extra qualifying traces mostly each cover the same full reference; no good reference requires assembling multiple source traces. This is consistent with almost-complete trajectories being present, but replay restart counts are not a guaranteed count of missing whole source fibers: replay starts from reference seeds, whereas these are independently seeded crop traces.

No inference, tracing, or BP was rerun. No constraints were used to choose matches. Previous spherical/BP-piece matching should not be used as the source-fiber equivalence result.
