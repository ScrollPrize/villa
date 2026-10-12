# Fresh state-LR and width-LR follow-up

## Protocol and selection

User requested another10xstateLR thenwidthfactor10x/.1x onbetterstateLRbyRefL1.
Widthfactor interpreted anddisclosed aswidthoptimizerLR, not lossweight.
Allfreshoriginalinitialization,positionlossweight0.5,bucketweight1,classbalance
perp:parallel10:1,fixedcharge32,allotherobjectiveandregularizersunchanged.
Samepositions5–11/IDs14–20,Z30k–50k,2199pieces,24771events,151hypotheses,
CCW,seed0,FP32,compiledpair/combinedcoordinates,sixexclusiveH100s0–5,
globalbatch24576.10080updates≈10000.649sampledpass equivalents (notexactepochs),
1008update diagnostics≈1kpasses. Kept existingpassprotocol for comparability.

StateLR0.7348469228349533 is10xcurrent0.07348469228349533; widthLR2unchanged.
FinalreferenceL1=5.41051 versusbaseline1.96746, so currentstateLR selected for
widthLR20/.2. Rejectionchangescohort; reportedretentionalso favorscurrentstateLR.
This is metric-basedselectiononthisregionalbenchmark,notgeneralizationproof.

## Commands

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_bucket_half_lr10_fresh.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_bucket_width_suite.json
```

Widthchildren:run_uv_bucket_width_{high,low}.json.
Outputs:
- data/uv_bucket_half_lr10_fresh_v1
- data/uv_bucket_width_high_v1
- data/uv_bucket_width_low_v1

Eachcontainsconfig/history/summaryJSONandsolution.npz. Usefreshoutputrootsfor
reruns,donotoverwritecompletedresults. No productioncode/defaultchanged.

## Final results (same objective; losses comparable)

| State LR | Width LR | Loss | Ref L1 | Within±2 % | Retained length % | Bucket sign % | Width factor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.07348469228349533 | 2 | 3.7064 | 1.9675 | 68.746 | 50.177 | 0.0000 | 0.9137 |
| 0.7348469228349533 | 2 | 7.5833 | 5.4105 | 62.605 | 44.575 | 0.0000 | 0.9833 |
| 0.07348469228349533 | 20 | 3.7045 | 1.9627 | 68.610 | 50.177 | 0.0000 | 0.9140 |
| 0.07348469228349533 | 0.2 | 3.7081 | 1.9769 | 68.568 | 50.165 | 0.0000 | 0.9141 |

HigherstateLR worsensbothlossandL1,despitezero bucketviolations. Width20/.2
haveonlyminimalimpact:width20L1gain~0.00475(~0.24%)vswidth2,notmeaningful
withoutrepeat/seedtesting. KeepcurrentstateLRandwidthLRdefaultsunchanged.

Lossismeanonlineobjectiveoverpreceding1kpasses,notfrozenstateevaluation.
RefL1andwithin2globalmedianoffsetcorrectedonretainedfibers; widthsmeanratio
fitted/referenceafteroffsetalignment. Signcolumnsbucket-only,notcoordinate
ordering. Rangequantilesandretentionareincludedtoexposecohort/driftchanges.

## 10x state LR / width LR2

| Pass equivalents | Loss | Ref L1 | Within±2 % | Retained length % | Perp sign % | Parallel sign % | Width factor | Fit q10/q90 | Ref q10/q90 | Global offset |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 0.00 | — | 3.1957 | 45.119 | 100.000 | 37.7750 | 23.8614 | 0.9829 | 11.0–23.0 | 15.0–20.0 | -1 |
| 1000.06 | 18.7821 | 2.3669 | 64.368 | 36.806 | 0.0000 | 0.0000 | 0.9140 | 13.0–22.0 | 14.0–20.0 | 0 |
| 2000.13 | 5.9670 | 2.4122 | 64.524 | 38.548 | 0.0000 | 0.0000 | 0.9139 | 13.0–22.0 | 14.0–20.0 | 0 |
| 3000.19 | 7.0271 | 3.0017 | 63.856 | 40.040 | 0.0000 | 0.0000 | 0.9129 | 13.0–23.0 | 14.0–20.0 | 0 |
| 4000.26 | 7.4551 | 3.5788 | 63.629 | 40.896 | 0.0000 | 0.0000 | 0.9135 | 13.0–23.0 | 14.0–20.0 | 0 |
| 5000.32 | 6.6260 | 4.0802 | 63.776 | 41.407 | 0.0000 | 0.0000 | 0.9136 | 13.0–23.0 | 14.0–20.0 | 0 |
| 6000.39 | 7.7458 | 4.4143 | 62.660 | 41.996 | 0.0000 | 0.0000 | 0.9127 | 13.0–23.0 | 14.0–20.0 | 0 |
| 7000.45 | 8.1045 | 4.8109 | 63.133 | 42.636 | 0.0000 | 0.0000 | 0.9130 | 13.0–23.0 | 14.0–20.0 | 0 |
| 8000.52 | 8.1126 | 4.9969 | 62.656 | 43.343 | 0.0000 | 0.0000 | 0.9112 | 13.0–23.0 | 14.0–20.0 | 0 |
| 9000.58 | 6.7823 | 4.9251 | 62.085 | 43.945 | 0.0000 | 0.0000 | 0.9118 | 13.0–23.0 | 14.0–20.0 | 0 |
| 10000.65 | 7.5833 | 5.4105 | 62.605 | 44.575 | 0.0000 | 0.0000 | 0.9833 | 12.0–23.0 | 14.0–20.0 | -1 |

## Baseline state LR / width LR20

| Pass equivalents | Loss | Ref L1 | Within±2 % | Retained length % | Perp sign % | Parallel sign % | Width factor | Fit q10/q90 | Ref q10/q90 | Global offset |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 0.00 | — | 3.1957 | 45.119 | 100.000 | 37.7750 | 23.8614 | 0.9829 | 11.0–23.0 | 15.0–20.0 | -1 |
| 1000.06 | 75.9450 | 2.0456 | 67.505 | 33.677 | 0.0000 | 0.0432 | 0.9492 | 13.0–22.0 | 14.0–20.0 | 0 |
| 2000.13 | 5.3596 | 2.0265 | 68.205 | 36.809 | 0.0000 | 0.0000 | 0.9174 | 13.0–22.0 | 14.0–20.0 | 0 |
| 3000.19 | 4.8954 | 2.0416 | 67.237 | 38.919 | 0.0000 | 0.0000 | 0.9130 | 13.0–22.0 | 14.0–20.0 | 0 |
| 4000.26 | 4.5955 | 2.0316 | 68.210 | 41.731 | 0.0000 | 0.0000 | 0.9130 | 13.0–22.0 | 14.0–20.0 | 0 |
| 5000.32 | 4.3385 | 2.0201 | 68.284 | 43.708 | 0.0000 | 0.0000 | 0.9136 | 13.0–22.0 | 14.0–20.0 | 0 |
| 6000.39 | 4.1589 | 2.0196 | 68.165 | 45.488 | 0.0000 | 0.0000 | 0.9133 | 13.0–22.0 | 14.0–20.0 | 0 |
| 7000.45 | 4.0126 | 2.0099 | 68.182 | 46.658 | 0.0000 | 0.0000 | 0.9147 | 13.0–22.0 | 14.0–20.0 | 0 |
| 8000.52 | 3.8976 | 1.9833 | 68.619 | 48.211 | 0.0000 | 0.0000 | 0.9143 | 13.0–22.0 | 14.0–20.0 | 0 |
| 9000.58 | 3.7900 | 1.9590 | 68.536 | 49.605 | 0.0000 | 0.0000 | 0.9134 | 13.0–22.0 | 14.0–20.0 | 0 |
| 10000.65 | 3.7045 | 1.9627 | 68.610 | 50.177 | 0.0000 | 0.0000 | 0.9140 | 13.0–22.0 | 14.0–20.0 | 0 |

## Baseline state LR / width LR0.2

| Pass equivalents | Loss | Ref L1 | Within±2 % | Retained length % | Perp sign % | Parallel sign % | Width factor | Fit q10/q90 | Ref q10/q90 | Global offset |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 0.00 | — | 3.1957 | 45.119 | 100.000 | 37.7750 | 23.8614 | 0.9829 | 11.0–23.0 | 15.0–20.0 | -1 |
| 1000.06 | 75.9458 | 2.0531 | 67.691 | 33.677 | 0.0000 | 0.0432 | 0.9139 | 13.0–22.0 | 14.0–20.0 | 0 |
| 2000.13 | 5.3600 | 2.0282 | 68.178 | 36.790 | 0.0000 | 0.0000 | 0.9139 | 13.0–22.0 | 14.0–20.0 | 0 |
| 3000.19 | 4.8946 | 2.0669 | 67.210 | 38.872 | 0.0000 | 0.0000 | 0.9139 | 13.0–22.0 | 14.0–20.0 | 0 |
| 4000.26 | 4.5955 | 2.0455 | 68.364 | 41.731 | 0.0000 | 0.0000 | 0.9139 | 13.0–22.0 | 14.0–20.0 | 0 |
| 5000.32 | 4.3389 | 2.0425 | 68.284 | 43.708 | 0.0000 | 0.0000 | 0.9138 | 13.0–22.0 | 14.0–20.0 | 0 |
| 6000.39 | 4.1598 | 2.0406 | 68.142 | 45.417 | 0.0000 | 0.0000 | 0.9138 | 13.0–22.0 | 14.0–20.0 | 0 |
| 7000.45 | 4.0122 | 2.0234 | 68.111 | 46.658 | 0.0000 | 0.0000 | 0.9138 | 13.0–22.0 | 14.0–20.0 | 0 |
| 8000.52 | 3.8992 | 1.9972 | 68.550 | 48.211 | 0.0000 | 0.0000 | 0.9139 | 13.0–22.0 | 14.0–20.0 | 0 |
| 9000.58 | 3.7927 | 1.9740 | 68.677 | 49.453 | 0.0000 | 0.0000 | 0.9140 | 13.0–22.0 | 14.0–20.0 | 0 |
| 10000.65 | 3.7081 | 1.9769 | 68.568 | 50.165 | 0.0000 | 0.0000 | 0.9141 | 13.0–22.0 | 14.0–20.0 | 0 |

## Runtime and validation

| Run | Solve s | Update mean/p50/p95 ms | Peak GiB/GPU |
|---|---:|---|---:|
| Baseline state LR / width LR2 | 172.009 | 16.686/16.658/16.940 | 1.159 |
| 10x state LR / width LR2 | 165.380 | 16.210/16.167/16.570 | 1.161 |
| Baseline state LR / width LR20 | 166.181 | 16.287/16.385/16.671 | 1.161 |
| Baseline state LR / width LR0.2 | 169.089 | 16.574/16.522/16.811 | 1.161 |

Singletrialeach,10080updates,first5excluded,maxrankupdatetimings;wall excludes
managerwait/startup. Independentconfigurationreview;9regressiontestspassed2.58s.
ProductioncompiledsixGPUgradient/coordinatechecks,allfinalarraysfinite,
bothrunnersexit0. Managers8a78b8b9-dbd5-457e-a9bc-0283fc51d4f2and
5d39fc6f-0124-4db5-8957-f0172effec43finishedwithheld0.

```bash
/home/hendrik/.venv_las/bin/python -m pytest -q --noconftest -o pythonpath=/home/hendrik/villa2/vesuvius/src vesuvius/tests/test_fiber_uv_bucket_split.py
```
