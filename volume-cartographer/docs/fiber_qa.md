# Fiber annotation QA

`scripts/fiber_qa.py` reads a directory of VC3D fiber annotations
(`vc3d_fiber` JSON, as saved by the line annotation tool and synced by
`vc_sync.py`) and lists the fibers worth a second look before they are
reviewed, published or used as training or evaluation references. It only
reads; it never changes a file.

```sh
python volume-cartographer/scripts/fiber_qa.py /data/fibers/PHerc0139 \
    --json fiber_qa.json
```

It needs Python 3 and numpy. Several directories (or single files) can be
given; fibers are matched by file name, as links are.

## Checks

| Check | Severity | What it reports |
| --- | --- | --- |
| `load` | error | What an unmodified VC3D refuses or rewrites on load: a control point that is not on the saved line, a link whose other fiber is missing, a link without its reciprocal entry, link positions or directions that disagree. Unreadable JSON and two files with the same name are reported here too. These are the checks of `scripts/fiber_loader_check.py`, an independent port of the loader's validation (also used by the `fiber_merge.py` tests). |
| `fold_back` | warning | The saved line reverses direction: the chord over the `--fold-window` units before a point and the chord over the same length after it point more than 120 degrees apart (`--fold-cos -0.5`). Usually a trace that overshot a control point and came back to it. |
| `retrace` | warning | The saved line runs back over its own path: points at least `--retrace-min-gap` units apart along the line lie within `--retrace-radius` units (plus half the median step) of each other with opposite directions (`--retrace-cos -0.8`). Reported once per fiber with the length traced twice and each stretch. A loop of the same fiber one winding later runs the same way, so it is not reported. |
| `jump` | warning | One step of the saved line is longer than `--jump-factor` times the fiber's median step and longer than `--jump-min` units. |
| `duplicate` | warning, or info when tagged | At least `--duplicate-fraction` of the shorter fiber's points lie within `--duplicate-radius` units (plus half the median step) of another fiber. `vc_sync.py hfsync` skips fibers tagged `possible_duplicate` by default; a pair that already carries the tag is listed as info. |
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
and every finding with the same fields. `--checks` runs a subset, for example
`--checks load` before a sync.

The exit code is 1 when there is an error finding (`--fail-on error`, the
default), or any warning (`--fail-on warning`), and 0 otherwise or with
`--fail-on never`; 2 for an unknown check name.

## Limitations

- The load check is a port of VC3D's validation, not the loader itself; a
  file it passes can still fail for a reason it does not model.
- Shape findings are candidates for review. A sharp but real turn of a fiber
  can be reported as a fold-back; look at the spot before changing anything.
- Duplicates are judged on geometry alone, so a shorter fiber traced along a
  longer one is reported even when both were intended; a crossing or an
  adjacent fiber is not, since only a small share of its points is close.
