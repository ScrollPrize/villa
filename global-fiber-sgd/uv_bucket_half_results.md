# Position weight0.5: matched10k trials

Only change versus uv_bucket_split_results.md: position_distance_weight0.1→0.5.
Bucket distance weight1; bucket-only sign and immutable seam correction unchanged.
All phase/offset/V/continuation/rejection terms, class balancing10:1, fixed charge32,
initializations and151hypotheses unchanged. StateLR0.07348469228349533,widthLR2.
Same positions5–11/IDs14–20,Z30000–50000,2199pieces/24771events,seed0,FP32,
compiled pair/combined coordinates. Globalbatch24576,6exclusiveH100GPUs0–5.
10080 sampled minibatchupdates =10000.649pass equivalents, not exhaustiveepochs.

## Command and outputs

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_bucket_half_suite.json
```

Configs run_uv_bucket_half_{fresh,reference}.json; outputs
`data/uv_bucket_half_{fresh,reference}_v1/{history.json,summary.json,solution.npz}`.
Use new output roots for a rerun; completed results are protected against overwrite.
Manager81363a56-ed6c-4634-86a7-3f50e6637e2e finished, no held resources.

Loss is mean online objective over preceding1kpasses, not a frozen-stateglobal
objective. Changing weights changes loss definition: compare trends withinruns,
not raw values acrossobjectives. Reference metrics use globalmedianoffset and
retainedfibercohort; retention matters. Bucket violations are not coordinate
signviolations. Widthfactor is mean fitted/reference afteroffsetalignment.

## fresh

| Pass equivalents | Loss | Ref L1 | Within±2 % | Retained length % | Perp bucket sign % | Parallel bucket sign % | Width factor | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 3.1957 | 45.119 | 100.000 | 37.7750 | 23.8614 | 0.9829 | 11.0–23.0 | 15.0–20.0 |
| 1000.06 | 75.9456 | 2.0559 | 67.505 | 33.677 | 0.0000 | 0.0432 | 0.9249 | 13.0–22.0 | 14.0–20.0 |
| 2000.13 | 5.3600 | 2.0359 | 67.921 | 36.790 | 0.0000 | 0.0000 | 0.9222 | 13.0–22.0 | 14.0–20.0 |
| 3000.19 | 4.8949 | 2.0661 | 66.884 | 38.872 | 0.0000 | 0.0000 | 0.9189 | 13.0–22.0 | 14.0–20.0 |
| 4000.26 | 4.5953 | 2.0409 | 68.056 | 41.731 | 0.0000 | 0.0000 | 0.9160 | 13.0–22.0 | 14.0–20.0 |
| 5000.32 | 4.3387 | 2.0336 | 68.284 | 43.708 | 0.0000 | 0.0000 | 0.9143 | 13.0–22.0 | 14.0–20.0 |
| 6000.39 | 4.1601 | 2.0232 | 68.215 | 45.458 | 0.0000 | 0.0000 | 0.9138 | 13.0–22.0 | 14.0–20.0 |
| 7000.45 | 4.0117 | 2.0135 | 68.182 | 46.658 | 0.0000 | 0.0000 | 0.9138 | 13.0–22.0 | 14.0–20.0 |
| 8000.52 | 3.8987 | 1.9951 | 68.619 | 48.211 | 0.0000 | 0.0000 | 0.9143 | 13.0–22.0 | 14.0–20.0 |
| 9000.58 | 3.7917 | 1.9699 | 68.446 | 49.529 | 0.0000 | 0.0000 | 0.9142 | 13.0–22.0 | 14.0–20.0 |
| 10000.65 | 3.7064 | 1.9675 | 68.746 | 50.177 | 0.0000 | 0.0000 | 0.9137 | 13.0–22.0 | 14.0–20.0 |

## reference

| Pass equivalents | Loss | Ref L1 | Within±2 % | Retained length % | Perp bucket sign % | Parallel bucket sign % | Width factor | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 0.0009 | 100.000 | 100.000 | 9.2860 | 2.0929 | 1.0000 | 15.0–20.0 | 15.0–20.0 |
| 1000.06 | 49.8866 | 0.5144 | 98.861 | 66.970 | 0.0000 | 0.0000 | 0.9732 | 14.0–20.0 | 14.0–20.0 |
| 2000.13 | 3.4170 | 0.5567 | 98.752 | 67.458 | 0.0000 | 0.0000 | 0.9334 | 14.0–20.0 | 14.0–20.0 |
| 3000.19 | 3.1648 | 0.5565 | 98.706 | 68.015 | 0.0000 | 0.0000 | 0.9073 | 14.0–20.0 | 14.0–20.0 |
| 4000.26 | 3.0536 | 0.5484 | 98.833 | 68.636 | 0.0000 | 0.0000 | 0.9000 | 14.0–20.0 | 14.0–20.0 |
| 5000.32 | 2.9587 | 0.5576 | 98.675 | 69.711 | 0.0000 | 0.0000 | 0.8970 | 14.0–20.0 | 14.0–20.0 |
| 6000.39 | 2.8511 | 0.5434 | 98.858 | 70.219 | 0.0000 | 0.0000 | 0.8953 | 14.0–20.0 | 14.0–20.0 |
| 7000.45 | 2.7704 | 0.5554 | 98.586 | 70.952 | 0.0000 | 0.0000 | 0.8944 | 14.0–20.0 | 14.0–20.0 |
| 8000.52 | 2.6991 | 0.6038 | 98.260 | 71.367 | 0.0000 | 0.0000 | 0.8940 | 15.0–20.0 | 14.0–20.0 |
| 9000.58 | 2.6386 | 0.6287 | 97.822 | 71.813 | 0.0000 | 0.0000 | 0.8958 | 15.0–20.0 | 14.0–20.0 |
| 10000.65 | 2.5882 | 0.6423 | 97.896 | 72.823 | 0.0000 | 0.0000 | 0.8951 | 15.0–20.0 | 14.0–20.0 |

## Final comparison

| Init | Formulation | Loss (different objectives) | Ref L1 | Within±2 % | Retained length % | Width factor |
|---|---|---:|---:|---:|---:|---:|
| fresh | Before split | 2.7149 | 1.2058 | 86.635 | 65.508 | 0.8514 |
| fresh | Position0.1 | 3.6809 | 2.1762 | 64.905 | 50.411 | 0.9123 |
| fresh | Position0.5 | 3.7064 | 1.9675 | 68.746 | 50.177 | 0.9137 |
| reference | Before split | 2.5073 | 0.6331 | 97.615 | 78.247 | 0.9082 |
| reference | Position0.1 | 2.5355 | 0.6235 | 98.618 | 73.071 | 0.9088 |
| reference | Position0.5 | 2.5882 | 0.6423 | 97.896 | 72.823 | 0.8951 |

0.5improvesfreshL1/within2 vs0.1butnotretention; stillworsethanpresplitbaseline.
Reference improvesearly butdriftsafter7k despitefallingloss; offsetremains0,
so this is not an offset-switchartifact. FinalreferenceL1/within2 andretention
are slightlyworsethan0.1. Zero finalbucketviolations inboth is not proof of
correctplacement. Keep experiment opt-in; no defaultobjectivepromotion.

## Timing / validation

| Init | Solve s | Update mean/p50/p95 ms | Peak GiB/GPU |
|---|---:|---|---:|
| fresh | 172.009 | 16.686/16.658/16.940 | 1.159 |
| reference | 171.213 | 16.788/16.723/16.928 | 1.161 |

Singletrial,10080updates,first5excluded,max-rankupdatetimes; wall excludesmanager
wait/processstartup.9split tests passed2.52s; both sixGPU compiled/eagergradient
checks andcoordinatevalue/gradient/bitwiserepeatpassed. Finalarraysfinite,
suiteexit0. Independentconfigreviewconfirmed onlyweight/root/labelchanges.

Test command:

```bash
/home/hendrik/.venv_las/bin/python -m pytest -q --noconftest -o pythonpath=/home/hendrik/villa2/vesuvius/src vesuvius/tests/test_fiber_uv_bucket_split.py
```
