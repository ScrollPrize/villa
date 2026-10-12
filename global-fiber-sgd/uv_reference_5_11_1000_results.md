# Positions5–11:1000-pass fresh/reference joint solves

Same audited reference-selected IDs14–20,baseZ30000–50000:2192originalfibers,
2199pieces,24771constraints,onecontinuation. No re-extraction or new labels.
Original fresh radial and reference-assignment/width initializations,not warm
resume. Full-parent radial prior unchanged;151hypotheses,CCW,compiled squared
pairloss,combinedcoordinates,all variables jointly trainable fromstart.

Six exclusive H100GPUs,FP32,globalbatch24576(4096/GPU),stateLR0.007348469228,
widthLR2,seed0.1008updates=1000.064915sampled dataset-pass equivalents;
101update reporting=100.204917passes. Exactpasscount isshown,notroundedepochs.
No objective/code changes from100passrun. Sampledwithreplacement as before.

ReferenceL1/within±2 use same globalL1-optimaloffset, equaloriginalfibers and
retained clipped-curve median assignments. Regionalwidthfactor compares only
fully reference-supported intervals14–20,after sameoffset. Rejection changes
the evaluated cohort. Loss is meanonlineobjective inprecedingreportinginterval,
not frozen-stateobjective; its100passaveraging differs fromearlier10passreports.
Finalinterval is99updates ratherthan101. Alltrajectorymetrics/commands below.

## Command

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_window1000_suite.json
```

Childconfigs:run_uv_window1000_{fresh,reference}.json.
Outputs:data/uv_window1000_w14_20_{fresh,reference}_v1,with config,history,
summary,solution; suite:data/uv_window1000_suite_v1. Existingcompletedoutputs
are protected; changeoutputroots for a fresh rerun. Visualization added below.

## Completed results

Both runs finished successfully. First101update reference metric dictionaries
exactly reproduce the corresponding100pass-run final metrics (loss averaging
window differs). CUDAcoordinatevalue/gradient/bitwise and6rankgradientpreflights
passed; finalstate/widtharraysfinite. No model/objectivechange.

Fresh: finalL1=1.522717,within2=80.9368%,sign1.5486%,retained95.2269%,loss2.53229.
Reference: finalL1=.580748,within2=98.1752%,sign.9449%,retained100%,loss1.59316.
Fresh steadily improves. Reference L1 initially worsens to.639 near500passes,
then improves to.581; nevertheless still worse than itsreferenceinitialstate.
Both objectives continue falling, so no convergence claim. Compared to100pass
final, reference signviolations4.5390%→.9449% while L1 .4831→.5807.
Fresh rejects4.7731%ofarc, so part of itsmetric improvement may be cohortchange.

## Timing

| Initialization | Solve wall s | Update mean ms | p50 ms | p95 ms | Peak GiB/GPU | Worker startup-through-summary s |
|---|---:|---:|---:|---:|---:|---:|
| fresh | 17.558 | 15.678 | 15.477 | 16.370 | 1.160 | 28.341 |
| reference | 18.184 | 16.333 | 16.277 | 16.513 | 1.160 | 29.377 |

Suiteelapsed95.933s including managerwait/workerstartup/preflight;
solvewall includes optimization/diagnostics/save,notstartup. Singletrial perrun,
first5CUDAupdates excluded; eachupdate maxacrossranks. GPUlease released after
bothsolves. No single-GPU speedup comparison implied.

## fresh: every ~100 passes

| Passes | Loss | Ref L1 | Within±2 % | Retained % | Sign violations % | Widthfactor | Offset | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 3.1957 | 45.119 | 100.000 | 37.662 | 0.9829 | -1 | 11/23 | 15/20 |
| 100.20 | 73.6229 | 2.7427 | 53.376 | 100.000 | 22.384 | 1.0010 | -1 | 12/22 | 15/20 |
| 200.41 | 38.5869 | 2.4015 | 60.538 | 100.000 | 14.514 | 0.9551 | 0 | 13/23 | 15/20 |
| 300.61 | 23.3690 | 2.1898 | 65.283 | 100.000 | 9.276 | 0.9738 | 0 | 13/22 | 15/20 |
| 400.82 | 15.2015 | 2.0178 | 69.571 | 100.000 | 6.213 | 0.9893 | 0 | 13/22 | 15/20 |
| 501.02 | 10.3737 | 1.8838 | 73.043 | 99.352 | 4.645 | 1.0021 | 0 | 13/22 | 15/20 |
| 601.23 | 7.3106 | 1.7792 | 75.161 | 98.472 | 3.606 | 1.0131 | 0 | 13/22 | 15/20 |
| 701.43 | 5.3187 | 1.6926 | 77.283 | 96.985 | 2.739 | 1.0223 | 0 | 13/22 | 15/20 |
| 801.64 | 4.0048 | 1.6359 | 78.665 | 95.869 | 2.360 | 1.0300 | 0 | 13/22 | 15/20 |
| 901.84 | 3.1293 | 1.5716 | 80.056 | 95.317 | 1.903 | 1.0365 | 0 | 13/22 | 15/20 |
| 1000.06 | 2.5323 | 1.5227 | 80.937 | 95.227 | 1.549 | 1.0420 | 0 | 13/22 | 15/20 |

## reference: every ~100 passes

| Passes | Loss | Ref L1 | Within±2 % | Retained % | Sign violations % | Widthfactor | Offset | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 0.0009 | 100.000 | 100.000 | 10.981 | 1.0000 | 0 | 15/20 | 15/20 |
| 100.20 | 52.7190 | 0.4831 | 100.000 | 100.000 | 4.539 | 1.0055 | 0 | 14/20 | 15/20 |
| 200.41 | 26.6304 | 0.5283 | 99.681 | 100.000 | 5.812 | 1.0078 | 0 | 14/20 | 15/20 |
| 300.61 | 15.7313 | 0.5958 | 98.403 | 100.000 | 3.818 | 1.0092 | 0 | 14/20 | 15/20 |
| 400.82 | 9.9925 | 0.6369 | 97.993 | 100.000 | 2.192 | 1.0095 | 0 | 14/20 | 15/20 |
| 501.02 | 6.7105 | 0.6391 | 97.993 | 100.000 | 1.683 | 1.0092 | 0 | 14/20 | 15/20 |
| 601.23 | 4.7097 | 0.6273 | 97.993 | 100.000 | 1.501 | 1.0086 | 0 | 14/20 | 15/20 |
| 701.43 | 3.4261 | 0.6214 | 98.175 | 100.000 | 1.195 | 1.0078 | 0 | 14/21 | 15/20 |
| 801.64 | 2.5769 | 0.6090 | 98.084 | 100.000 | 1.256 | 1.0068 | 0 | 14/20 | 15/20 |
| 901.84 | 1.9965 | 0.5935 | 98.084 | 100.000 | 1.010 | 1.0058 | 0 | 14/20 | 15/20 |
| 1000.06 | 1.5932 | 0.5807 | 98.175 | 100.000 | 0.945 | 1.0046 | 0 | 14/20 | 15/20 |

Widthfactorfresh changes between100and200partly because referenceoffset changes
-1→0; comparison now uses different fitted/reference intervalpairs. It is not
solely physicalwidthchange. Regional meanwidthfactor near1 can still hide
individualwidtherrors; reference final meanabsolute relativeerror9.3062%.

## Visualization

Existing CPUlayout-onlyrenderer; map-only1000pxhighLZW TIFF,equalU/V scale
within each map,SVG/vectorPDF with labels/histograms,standalone histograms.
Hblue,Vorange,opacity=jointbestwindingprobability. Dashedfittedwindingboundaries.
Eachmap's bounds enclose itscompletefittedgeometry; they do not cut toGTZbounds
because V is optimized andmaydeviate. No newreferenceUVprojection wasgenerated.

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_window1000_vis.json
```

- data/uv_window1000_vis_v1/fresh/layout.tif —9918×1000.
- data/uv_window1000_vis_v1/reference/layout.tif —5770×1000.
- Eachdirectoryalso haslayout.svg,layout.pdf,histograms.tif,histograms.svg,
  andsummary.jsonwith sourcehashes,bounds,confidence statistics.

BothTIFFheadersverifiedLZW/height1000. BothSVG/PDFandhistograms exist; noPNGs.
