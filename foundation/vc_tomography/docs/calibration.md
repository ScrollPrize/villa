# Calibration and alignment

Before a reconstruction can assign a raw pixel value to the line through the sample it measured, two things must
be known:

- **Geometry**: for every frame of every scan, where the detector pixels look into the sample. This includes the
  detector's distortion, each scan's rotation axis, height and position, and the time at which each detector row
  was exposed.
- **Intensity**: the dark and flat images that convert raw counts to transmission, and the weights that blend
  overlapping scans.

Both are measured from the scans' own frames. Nothing has to be entered by hand, except the camera's row time
when the data do not contain it (see [Row time](#row-time)).

`vc_tomography_reconstruct` runs both steps on first use and caches the results. Run them yourself with
`vc_tomography_calibration` to inspect the results, to reuse one geometry for several reconstructions, or when the
acquisition needs options that the automatic run does not set.

The order is fixed: geometry first, then intensity, because the intensity step lays out the merged frame from the
geometry.

```bash
vc_tomography_calibration geometry  HA.h5 TA.h5 --out CAL/geometry.json
vc_tomography_calibration intensity HA.h5 TA.h5 --geometry CAL/geometry.json --out CAL
vc_tomography_reconstruct PROJ --cache_dir CAL --z 0 64 --tag test
```

Scans are given as master files (or scan directories, for `intensity`). The scan with the rotation axis inside
its field of view comes first; the other scans follow in the order they lie across the beam.

## Geometry calibration

`vc_tomography_calibration geometry` writes one JSON file, `geometry.json`.

### What it measures

All parameters are solved together in one least-squares problem (Ceres, with a robust Huber loss):

- **Detector map**: a third-order polynomial displacement of every pixel (lens distortion). `--order 0` disables it.
- **Per scan**: the rotation axis column, its slope along the vertical travel, the axial scale (the actual helical
  pitch relative to the recorded one), and the scan's position and height relative to the first scan (the anchor).
- **Per scan, after the main solve**: the small tilt of the sample between scans (tens to hundreds of
  microradians). The reconstruction applies the part of it about the beam direction.
- **Detector roll**: the rotation of the detector about the beam.
- **Rotation stage wobble**: the sample's height once per turn and the axis column twice per turn, shared by all
  scans on the stage.
- **Row time**: the time between the exposure of consecutive detector rows (rolling shutter).
- **Beam divergence**: the angle between rays at neighbouring columns, when the record gives the source distance.

### How it measures

1. **Layout.** From the acquisition record and a few frames: frames per turn, which stretches of frames are on
   disk, the direction of the rows relative to the vertical stage, a coarse axis column per scan, and the rows
   that hold the sample mount (which are excluded).
2. **Row time.** Fitted to the intensity dips of a storage-ring refill, if one happened during a stretch on disk.
3. **Comparisons.** Blocks of frames that record the same lines through the sample on different parts of the
   detector are registered against each other to a fraction of a pixel. These are: opposite views (half a turn
   apart) within the rest turns and the helix, the helix against itself one turn later, rest turns against the
   helix, and one scan against its neighbour at the same angle where their columns overlap. Blocks are at most
   512 columns wide and are cut into patches of 128 × 128 pixels (`--patch`).
4. **Solve.** Every patch match gives two rays that should coincide. The solver finds the parameters that bring
   them together. What the model cannot explain stays in the residuals.
5. **Report.** Residuals per scan and per kind of comparison, and diagnostics (below).

Registration runs on CUDA when a GPU is present (`--device cpu|cuda|auto`).

### What it needs from the acquisition

- At least one scan with the rotation axis inside its field of view.
- Either a helix whose travel per turn is less than the detector height, or circular scans at several heights
  (a grid). Rest turns before and after the helix are used if they are on disk but are not required.
- For the row time: a ring refill during a stretch on disk, a grid read from the middle row outward
  (`--rows-from-middle`), or the value given with `--row-time-us`.

### Commands by scan type

#### BM18 helical pair

```bash
vc_tomography_calibration geometry HA.h5 TA.h5 --out geometry.json
```

On PHerc0343P (two scans of 48 000 frames, RTX 5090) this takes 2 minutes 20 seconds with the frames in the page
cache and 5 minutes without, and solves 33 unknowns from 9858 patches.

#### BM18 multi-column scan

```bash
vc_tomography_calibration geometry HA.h5 TA.h5 TA2.h5 TA3.h5 --out geometry.json --row-time-us -6.46
```

Each scan is compared with its neighbours across the columns they share. A scan whose rotation axis lies outside
its field is first placed coarsely against its neighbour. When the overlaps are narrow, some terms of the
detector map cannot be told apart from the scans' axis columns and heights. Such terms (correlation above 0.95
with another unknown) are held at zero and listed in `solution.undetermined`.

#### Diamond I12 grid of circular scans

```bash
vc_tomography_calibration geometry r1c1.nxs r1c2.nxs r1c3.nxs r2c1.nxs r2c2.nxs r2c3.nxs \
    --out geometry.json --rows-from-middle --radial-map
```

- `--rows-from-middle`: the PCO.edge camera reads its two halves at once, from the middle row outward. With this
  option the row time is measured from the overlaps between scans at different heights.
- `--radial-map`: the scans overlap only along the frame edges, which cannot separate every polynomial term of
  the detector map. This option fits the map as a single radial lens distortion instead (one unknown).
- A circular scan does not travel along the axis, so its axial scale and axis slope are held fixed.
- Only the rows lit by the beam (above 0.12 of the brightest row's mean flat) are used.

#### BM18 mosaic tiles

```bash
vc_tomography_calibration geometry T0.h5 T1-01.h5 T1-02.h5 ... --out geometry.json --order 0 --row-time-us 0
```

This calibrates each tile from its own views only (axis, axial scale, wobble). The tiles' positions relative to
each other are not solved by this command. The Python functions `geometry.plan_tiles`, `geometry.measure_tiles` and
`geometry.solve_geometry` (with lens patches from overlapping tile volumes) measure them, but this is not yet a
step of the tool.

### Options

| Option | Meaning |
|---|---|
| `--out FILE` | output file (required) |
| `--row-time-us US` | row time in microseconds per stored row, signed. `0` means unknown: solve without it. If omitted, it is measured from a ring refill. |
| `--rows-from-middle` | the camera reads both halves at once from the middle row |
| `--order N` | polynomial order of the detector map (default 3; 0 for none) |
| `--radial-map` | fit the map as one radial distortion |
| `--patch N` | patch size in stored pixels (default 128) |
| `--reduce 1\|2` | measure on frames at 1/2 or 1/4 resolution |
| `--projections STORE.zarr` | read the frames, darks and flats from a projection store |
| `--threads N`, `--device cpu\|cuda\|auto` | compute resources |

### Row time

The detectors used read their rows one after another, so each row of a frame is exposed at a slightly different
rotation angle. Ignored, this rotates each frame slightly from its first row to its last: on the Paris4 mosaic,
by 0.9 to 1.6 px at the edge of a tile's field.

The row time is measured from a ring refill when one falls in the data. Otherwise give it with `--row-time-us`
(or `--set row_time_us=` on `vc_tomography_reconstruct`). It is a property of the camera's configuration, not of
the scan, so a value measured on one scan holds for other scans recorded with the same settings. The sign gives
the reading direction. The value is per stored row, whatever `--reduce` is.

### Reduced resolution

With `--reduce 1` or `--reduce 2` the step reads frames at a half or a quarter of the detector's resolution: from
the JPEG 2000 files' own lower resolutions or from a projection store. The result is still written in stored
(full-resolution) pixels. The patches cover the same area of the detector, so they contain fewer pixels.

A reduced calibration is consistent between scans but carries a small common offset. On seven Paris4 tiles,
compared with a full-resolution calibration:

| | Axis column | Axial scale | Run time |
|---|---|---|---|
| 1/2 | 0.07 to 0.09 px lower | 2 to 18 ppm lower | 2 min 24 s (full resolution: 4 min 18 s) |
| 1/4 | 0.21 to 0.24 px lower | 190 to 230 ppm lower | 59 s for all 19 tiles from a store |

The tiles relative to each other agree with the full-resolution result to 0.006 px (1/2) and 0.012 px (1/4).
The step does not run at 1/8: the overlap strips become too narrow.

### The output file

`geometry.json` contains:

| Key | Content |
|---|---|
| `scans` | the master files the calibration was made from. The reconstruction refuses a calibration of other scans. |
| `layout` | detector size, frames per turn, pixel size, fan, lit rows, and per scan its stretches, coarse axis and place |
| `row_time` | seconds per row and how it was found |
| `map` | the detector map's polynomial coefficients |
| `parameters` | every solved parameter, by name |
| `solution` | how the solve ended, residuals per kind of comparison, held (`undetermined`) and strongly correlated unknowns |
| `report` | per scan: what is left of its opposite views (mean, once and twice per turn) and the row time those views imply; spread of axis, slope and axial scale over the scans; per pair of scans: residuals in frames, rows and columns |
| `comparisons` | number of blocks, patch size, accepted patches per kind |

### Checking a calibration

- Residuals in `solution` and `report` should be a small fraction of a pixel. On PHerc0343P the rows between the
  two scans are left at 0.088 px rms.
- Reconstruct each scan alone (`--set only_scan=HA`, `--set only_scan=TA`, ...) and register the volumes against
  each other. On PHerc0343P the second scan lies 0.08 to 0.10 px (median) from the first.
- Compare the seams of a merged volume.

### Python

```python
from vc_tomography import geometry
cal = geometry.calibrate_geometry([ha_h5, ta_h5])
geometry.save(cal, "geometry.json")
```

The stages are also available one by one (`_core.CalibrationSession`, `_core.geometry_read_acquisition`,
`geometry.measure_comparisons`, `geometry.solve_geometry`, `geometry.report`), and `solve_geometry` can hold or
free unknowns by name (`held=`, `freed=`).

## Intensity calibration

```bash
vc_tomography_calibration intensity HA TA [TA2 ...] --geometry geometry.json --out CAL
```

The scans are master files or scan directories, in the order of the geometry calibration. The step computes:

- the darks and flats of every scan, each flat scaled to the nominal ring current of 0.2 A, reduced as
  the ESRF reference (nx2etf) reduces them;
- the weight map of each scan, with a taper at every edge that overlaps another scan, and cropped to the lit rows;
- the detector's blur field, measured from the noise in the main scan's flats (used by
  `sharpness_weighting=auto`);
- the placement of every scan in the merged frame, from the geometry.

It writes `calibration_arrays.bin` and `calibration.json` to `CAL`. Use `CAL` as the reconstruction's
`--cache_dir`, and put `geometry.json` in the same folder (or pass `--set geometry=FILE`). The cache is keyed
by a fingerprint of the acquisition; an existing cache of the same acquisition is reused, and only the parts that
changed are recomputed.

`calibration_arrays.bin` is a single line of JSON naming each array, its shape and byte offset, followed by the
float32 little-endian values.

## Checking the acquisition record

`vc_tomography_calibration check-record` checks what a scan's BLISS master states, against itself and against the
raw frames. It changes nothing.

```bash
vc_tomography_calibration check-record HA.h5 TA.h5 --no-frames        # the record alone; reads no frames
vc_tomography_calibration check-record HA.h5 TA.h5 --out DIR          # all checks
```

| Check | What it does |
|---|---|
| `record` | frame counts, the map from frames to files, trigger timer, clocks, angles and heights against the raw encoder counts, commanded against recorded travel, the scans against each other |
| `inventory` | reads every frame and compares its sum, minimum and maximum with the statistics the detector software recorded |
| `series` | the dark and flat frames |
| `beam` | the brightness of the air beside the sample against the ring current |
| `views` | registers blocks that recorded the same lines (opposite views, rest turns against the helix, one scan against the other) |

`--checks LIST` selects a comma-separated subset. Each finding is `verified`, `fails`, `unverified` or `note`. The
position target is 0.01 px (`--position-target`); any accepted block beyond it fails the check. Results go to
`DIR/report.txt`, `report.json` and `views.tsv`. The exit status is 0 when the report was written, whatever its
findings. NXtomo records (Diamond I12) are not checked.
