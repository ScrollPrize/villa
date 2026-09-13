# Contrastive embedding consistency — 2026-09-13

## Snapshot and protocol

Current raw snapshot of `combined_5scroll_s1a_contrastive_128_sd1_16x8_ema_20260913_121417`: **2100 updates / 268,800 samples**, SHA256 `46ff8c8320a7787d1d4209fe0dbf8b9e3aea8f1e40a2314b9be708ded7489eb6`. Frozen copy and raw results: `/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_consistency_20260913_contrastive_v1/`.

Compared with [previous pairwise analysis](embedding_consistency_20260913.md): update 5300 / 678,400 samples. This is not a matched-training-age ablation.

Verified all ten case identities, sampled GT coordinates and handoff locations exactly match the previous run. Only checkpoint/output settings changed. Two held-out JSON fibers per scroll; centered 512-base-voxel arcs at 4-base-voxel spacing. Input level 1, 256³ tiles, overlap 64, border 16, output stride 4, hypothetical storage chunks 32. BF16 model, native CT reader, FP32 signed-vector blending followed by trilinear querying and normalization.

## Off-GT discrimination

Controls remain >80 base voxels from the full GT polyline. They are unlabeled other-fiber/background points, not the exact normal-ellipse training distribution. Presence ≥0.5 defines high presence. That subset changes with the model, so counts are shown. Across all 7,899 controls, no new cosine exceeded 0.9; the high-presence subset contains 1,024 controls.

| Scroll | Old high-presence n | Old cosine >0.9 | New high-presence n | New cosine >0.9 | New cosine p05 / median / p95 |
|---|---:|---:|---:|---:|---|
| PHerc0139 | 154 | 31.8% | 292 | 0% | -0.1891 / 0.1847 / 0.5163 |
| PHerc0332 | 215 | 28.4% | 245 | 0% | -0.3628 / 0.0283 / 0.4650 |
| PHerc1299 | 142 | 24.6% | 143 | 0% | -0.3537 / 0.0799 / 0.4754 |
| PHerc1451 | 96 | 26.0% | 144 | 0% | -0.3075 / 0.1374 / 0.6287 |
| PHercParis4 | 209 | 23.4% | 200 | 0% | -0.3079 / 0.1342 / 0.5138 |

## GT and boundary cosine

Adjacent comparisons use the same 4-base-voxel arc separation. Overlap means the **same point** inferred independently by overlapping tiles; it is not an adjacent-point measurement. Switches are dominant-contributor handoffs within a continuous blend. Storage crossings do not test serialization.

| Scroll | Same-point overlap | Within tile | Blended interior | Blended switch | Storage crossing | Switch pairs |
|---|---:|---:|---:|---:|---:|---:|
| PHerc0139 | 0.864334 | 0.935108 | 0.968996 | 0.987810 | 0.970528 | 4 |
| PHerc0332 | 0.817542 | 0.901180 | 0.959065 | 0.912804 | 0.959724 | 2 |
| PHerc1299 | 0.918319 | 0.940516 | 0.946077 | 0.900026 | 0.969577 | 2 |
| PHerc1451 | 0.970721 | 0.966074 | 0.967273 | 0.998173 | 0.990542 | 3 |
| PHercParis4 | 0.923691 | 0.935841 | 0.944193 | 0.966124 | 0.975034 | 4 |
| Equal-scroll mean | 0.898922 | 0.935744 | 0.957121 | 0.952987 | 0.973081 | — |
| Previous equal-scroll mean | 0.999994 | 0.999983 | 0.999986 | 0.999996 | 0.999984 | — |

## Longer GT separation

| Scroll | Within tile at 128 base voxels | Blended interior | Blended switch |
|---|---:|---:|---:|
| PHerc0139 | 0.912437 | 0.936821 | 0.883030 |
| PHerc0332 | 0.915549 | 0.956059 | 0.923152 |
| PHerc1299 | 0.888335 | 0.903306 | 0.895592 |
| PHerc1451 | 0.923641 | 0.915074 | 0.948970 |
| PHercParis4 | 0.928680 | 0.932721 | 0.925161 |

## Blending cancellation

Cancellation is ||Σw e|| / Σw||e|| at all eight query support corners, including corners with zero interpolation coefficient. A low corner value does not necessarily mean cancellation at the actual query. No zero GT vectors occurred.

| Scroll | Mean cancellation ratio | Worst corner |
|---|---:|---:|
| PHerc0139 | 0.913579 | 0.457243 |
| PHerc0332 | 0.811251 | 0.482394 |
| PHerc1299 | 0.976098 | 0.610204 |
| PHerc1451 | 0.966990 | 0.692282 |
| PHercParis4 | 0.962756 | 0.672535 |

## Interpretation

- Stronger local separation: the former near-identical off-GT embeddings are absent at the >0.9 threshold in this pilot. This is not proof that all neighboring fibers are separable.
- The old almost-perfect GT consistency was not evidence of useful identity: its GT directions were nearly identical across windows. The new representation is more variable both along GT and between tile contexts.
- Same-point overlap mean 0.899 (versus 0.999994) exposes cross-tile dependence. Some overlap comparisons are negative (minimum approximately −0.137). Local patch supervision alone does not establish a globally consistent embedding.
- Blended interior and handoff adjacent cosines are similar in the equal-scroll aggregate (0.957 versus 0.953), but only 15 switch pairs were measured. PHerc0332 and PHerc1299 have visibly lower switch cosines. Do not claim seam invariance.
- Blending remains nonzero, but cancellation has increased. Test a tracer with these embeddings before treating discrimination as a demonstrated tracing improvement.

## Reproduce

### Cosine histograms

CPU-only plots and numeric histograms were generated from these saved results:

- [PNG](/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_consistency_20260913_contrastive_v1/cosine_histograms.png)
- [SVG](/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_consistency_20260913_contrastive_v1/cosine_histograms.svg)
- [Counts, quantiles and bins](/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_consistency_20260913_contrastive_v1/cosine_histograms.json)

Positives use all distinct unordered pairs of GT points within each saved
blended window (82,560 pairs). Negatives are the saved same-tile unblended
controls: 7,899 total, 1,024 with presence>=0.5. Each distribution is normalized
separately; the overall panel averages normalized distributions across scrolls.
The blue positive distribution and orange high-presence negative distribution
have a binned overlap coefficient of 2.73% (sum of minimum probability per bin,
bin width0.05). This is descriptive histogram overlap, **not an error rate** or
a matched-anchor gap statistic. Pair samples are correlated along each fiber.

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment embedding-histograms
```

Five CPU tests passed, including distinct pairs, presence filtering, zero-vector
rejection, bin conservation and the existing interpolation/consistency tests.
Training was not interrupted for histogram generation.

Training automatically resumed after analysis; update 2200 (281,600 samples)
was observed with the original run identity.

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment embedding-consistency
```

Existing approved runner, one manager-exclusive GPU, 48 tile forwards; measured analysis time **34.5s**, excluding manager waiting and model/checkpoint startup. No training or tracer code changed. No F16/Zarr storage roundtrip, full 50-fiber trace evaluation, or non-CUDA platform validation was performed.
