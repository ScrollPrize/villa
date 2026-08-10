# Physical evaluation harness for surface predictions

Evaluates any binary surface prediction volume against ground-truth labels derived from a second, higher-resolution scan of the same scroll, rather than from another model or from mesh-derived annotations.

Two label volumes exist so far, built by registering the scroll pairs that have two public scans (split-half validated at 4.1 um and 2.4 um median error):

- PHerc0139: 9.362 um frame, truth from the 1.129 um scan
- PHerc1203 (Grand Prize list): 9.362 um frame, truth from the 2.403 um scan, including compressed and boundary-poor tissue

The label volumes (uint8 bit flags: valid / material / centerline / recto_band, plus boundary_poor for PHerc1203), the registration transforms, and the audit built on them are at
https://github.com/7jycwjmbfn-eng/pherc0139-physical-audit (labels attached to the release).

## Usage

```bash
python3 eval_surface_pred.py LABELS.zarr PRED.zarr PRED_LEVEL
```

`PRED_LEVEL` is the pyramid level of the prediction volume (0 or 1); level 0 is max-pooled to the label grid. Dependencies: numpy, scipy, numcodecs.

Reported metrics, each alongside a shifted-null control (the same prediction displaced 1.2 mm in-plane):

- point recall at 19 / 37 / 56 um from truth sheet centerlines
- arc-level recall and fully-missed rate over 1.2 mm sheet stretches
- recto-side overlap ratio

The null control matters in dense tissue: on PHerc1203 the null alone reaches 64.6 percent point recall at 37 um, so radius metrics without it overread performance exactly where the hard failures live.

Checked against the published m7 volume for PHerc0139, the harness reproduces the audit numbers in the repository above to four decimals from the packaged files alone.
