# Fiber annotation QA

`scripts/fiber_qa.py` lists the fibers worth a second look before they are
reviewed, ingested into an object store, or used as training or evaluation
references. It reads native fiber JSON (the `vc3d_fiber` files that
`vc_object_ingest --fiber` takes and `vc_object_export` writes, or a
`vc3d_fiber_collection` file), or every active fiber of an object store. It
only reads; it never changes a file or a store.

```sh
# A folder of native fiber JSON, for example before vc_object_ingest
python volume-cartographer/scripts/fiber_qa.py /data/native-fibers \
    --json fiber_qa.json

# Every active fiber in a store (opened read only)
python volume-cartographer/scripts/fiber_qa.py --store /data/fibers.vcobj
```

It needs Python 3 and numpy; `--store` also needs the `vc_object_store`
Python package. It reads each active fiber's current `line_points`,
`control_points` and control point tags straight from the store, so the
findings match those for the same fibers' files.
Several folders and files can be given together, with or without `--store`;
a fiber given twice (same UUID, or same file name for files without one) is
checked once and reported.

## Checks

| Check | Severity | What it reports |
| --- | --- | --- |
| `read` | error | A file that is not readable JSON, a fiber without `line_points` or `control_points`, or the same fiber given twice. Links between fibers are not checked here; `vc_object_ingest` checks them when fibers enter a store. |
| `fold_back` | warning | The saved line reverses direction: the chord over the `--fold-window` units before a point and the chord over the same length after it point more than 120 degrees apart (`--fold-cos -0.5`). Usually a trace that overshot a control point and came back to it. |
| `retrace` | warning | The saved line runs back over its own path: points at least `--retrace-min-gap` units apart along the line lie within `--retrace-radius` units (plus half the median step) of each other with opposite directions (`--retrace-cos -0.8`). Reported once per fiber with the length traced twice and each stretch. A loop of the same fiber one winding later runs the same way, so it is not reported. |
| `jump` | warning | One step of the saved line is longer than `--jump-factor` times the fiber's median step and longer than `--jump-min` units. |
| `duplicate` | warning, or info when tagged | At least `--duplicate-fraction` of the shorter fiber's points lie within `--duplicate-radius` units (plus half the median step) of another fiber. A pair where either fiber carries the `possible_duplicate` tag is listed as info, so pairs already reviewed stay quiet. |
| `ends` | info | A first or last control point without a `kollesis_termination` or `break` tag, that is, an end with no recorded reason. Listed only with `--show-info`; counted in the summary. |

Distances are in the files' own coordinate units, the voxels of the volume the
fibers were drawn on. The defaults suit fibers saved by the native tracer at
level 0 of a 2.4 um scan (about 8 units between saved points); every
threshold is a command-line option.

## Output

The text report starts with a count per check, then one line per error and
warning with the file, the line point, its arclength, the nearest control
point and its position, so the spot can be found in VC3D:

```text
620 fibers
  fold_back: 72 findings in 48 fibers
  retrace: 26 findings in 26 fibers
WARNING fold_back dj_..._000041.json: line reverses direction (chord cosine -1.00) at line point 2099 (arclength 16750.2, near control point 18) [19250.84, 15179.82, 32447.55]
```

`--json PATH` (or `-` for standard output) writes the summary, the options
and every finding with the same fields; a fiber that has a display name
(`name`) carries it too, since exported files are named by UUID. `--checks`
runs a subset of `shape`, `duplicate` and `ends`, for example `--checks shape`.

The exit code is 1 when there is an error finding (`--fail-on error`, the
default), or any warning (`--fail-on warning`), and 0 otherwise or with
`--fail-on never`; 2 for an unknown check name or no input.

## Limitations

- It does not check links between fibers; `vc_object_ingest` does, when
  fibers enter a store.
- Shape findings are candidates for review. A sharp but real turn of a fiber
  can be reported as a fold-back; look at the spot before changing anything.
- Duplicates are judged on geometry alone, so a shorter fiber traced along a
  longer one is reported even when both were intended; a crossing or an
  adjacent fiber is not, since only a small share of its points is close.
