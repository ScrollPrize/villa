# Physical evaluation harness for surface predictions

Evaluates any binary surface prediction volume against ground-truth labels derived from a second, higher-resolution scan of the same scroll, rather than from another model or from mesh-derived annotations.

Two label volumes exist so far, built by registering the scroll pairs that have two public scans (split-half validated at 4.1 um and 2.4 um median error):

- PHerc0139: 9.362 um frame, truth from the 1.129 um scan
- PHerc1203 (Grand Prize list): 9.362 um frame, truth from the 2.403 um scan, including compressed and boundary-poor tissue

The label volumes (uint8 bit flags: valid / material / centerline / recto_band, plus boundary_poor for PHerc1203, which flags material whose gap visibility falls under 0.40 and covers 72.8 percent of material voxels there rather than a small minority), the registration transforms, and the audit built on them are at
https://github.com/7jycwjmbfn-eng/pherc0139-physical-audit (labels attached to the release).

## Usage

```bash
python3 eval_surface_pred.py LABELS.zarr PRED.zarr PRED_LEVEL
```

`PRED_LEVEL` is the pyramid level of the prediction volume (0 or 1); level 0 is max-pooled to the label grid. Dependencies: numpy, scipy, numcodecs.

Reported metrics, each alongside a shifted-null control (the same prediction displaced 1.2 mm in-plane):

- point recall at 19 / 37 / 56 um from truth sheet centerlines
- arc-level recall and fully-missed rate over 1.2 mm sheet stretches
- `side_inward`, the fraction of centerline points where the prediction sits on the inward (recto) side of the sheet, with both its shifted null and an ideal-recto ceiling built from the truth (`side_inward_null`, `side_inward_ideal`, `side_skill_of_ideal`)

One metric stands outside that rule. `recto_side_ratio` is a raw mass-overlap quantity, prediction mass on the recto band against the verso side, and it carries no control; it is kept for continuity and is not the side metric the audit quotes. The estimator behind `side_inward` self-tests on synthetic stripes at four angles before any volume is read.

The side arms are conditioned, which matters if you plan to freeze against the calibrated skill. `side_inward` keeps coherent centerline points with the prediction within 3 voxels. `side_inward_null` and `side_inward_ideal` keep the subset of *those* points that also has their own band within 3 voxels, so the populations are real, real AND null, real AND ideal, and the calibration is a within-real comparison. This reproduces what the audit pass does. The `_full` variants of both controls drop the conditioning and select from the whole coherent centerline set; each arm reports its own decided count. `side_inward` is identical under either reading, since the real arm is the same set both ways, so un-nesting moves the controls only.

The null control matters in dense tissue: on PHerc1203 the null alone reaches 64.6 percent point recall at 37 um, so radius metrics without it overread performance exactly where the hard failures live.

Run against the published m7 volumes, the harness output is committed in the repository above at `results/eval_selfcheck_0139.json` and `results/1203/eval_selfcheck_1203.json`, so a rerun can be diffed against a file. The point and arc numbers match the audit to four decimals, as does the side null; the inward fraction and ceiling agree to three. The harness takes sheet normals from the packaged material mask because the label package does not carry the high-resolution grayscale the audit used, which moves the decided count by 0.09 percent on PHerc0139.

Label tarball hashes:

```
labels0139_L1.tar  sha256 42fe53b760c2c9347d9f215bafa68beec8e96121d03549dab56a52a9a0a9e8dd
labels1203_L1.tar  sha256 32a09f6081342b0f015b258ec577d0296ff23a55892af9785689d8a55bff344c
```
