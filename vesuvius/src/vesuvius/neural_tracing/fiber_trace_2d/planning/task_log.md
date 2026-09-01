# Task log: retune hard continuation and alignment falloff

## Starting point

- Starting revision: `582e27319`.
- Release binary: `volume-cartographer/build/bin/vc_fiber_trace_chunk`.
- 1024 tuning input: `data/workdir3/crop_traces.zarr`.
- 2048 held-out input:
  `data/workdir3/fiber-crop-2048/crop_traces.zarr`.
- Common settings: hard edge-local continuation, quality fraction `0.25`,
  piece length `512`, fixed phase `0.5`, fixed scale `0.822`, fixed
  orientation, both sign classes, parallel cutoff `0.5`, and 500 messages.
- Historical hard/off diagnostic anchor (invalid for the current selection):
  - 1024: converged, 8/8 exact references, 1,313/2,175 right constraints,
    1,357 active and 3 Defect pieces.
  - 2048: converged, 6/8 exact references, 551/952 right constraints, 2,488
    active and 35 Defect pieces.

## Plan review

Independent review required these corrections before execution:

- fixed eligible-reference denominators and predeclared 90% reference/piece
  coverage gates so Defect-heavy candidates cannot win by abstention;
- finite absolute parameter grids, a complete-tuple cache, deterministic
  best-improvement ordering, and a 500-scenario guard per family;
- explicit zero class-weight and zero sign-cost candidates;
- complete coordinate-block repetition after every accepted move;
- mandatory hard/30-degree anchors after the user's clarification;
- treatment of overlapping 2048 as frozen larger-context validation, not
  independent held-out GT;
- three rotated timing repetitions and full prepass/tuple logging.

All corrections are incorporated in `task_plan.md`. The subsequent user
clarification fixes hard signs at 30 degrees for every valid scenario and
defers all 2048 runs.

## Results

All scenarios used Release revision `582e27319`, the 1024 crop, hard
continuation, both hard signs fixed at 30 degrees, fixed phase `0.5`, fixed
scale `0.822`, piece length `512`, quality fraction `0.25`, parallel winding
cutoff `0.5`, and 500 message iterations. No 2048 scenario was run.

The weight order below is `perp_0.5,perp_1.5+,parallel_0,parallel_1,parallel_2+`.

| Normal confidence | State | Decision | Weights | Sign | Defect | Temp | Exact | Right/evaluated | Active/Defect | All infringed/active |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | anchor | legacy | `8,1,2,2,1` | 44 | 100 | 1.25 | 7/8 | 1175/2050 | 1294/66 | 24075/62615 |
| none | tuned | linear | `8,1,1,2,1` | 44 | 100 | 1.25 | 8/8 | 1279/2114 | 1321/39 | 23896/64653 |
| linear | anchor | legacy | `8,1,2,2,1` | 44 | 100 | 1.25 | 5/8 | 1035/2100 | 1309/51 | 25869/64656 |
| linear | tuned | cosine | `0,2,2,2,1` | 44 | 100 | 1.25 | 8/8 | 1369/2100 | 1329/31 | 23863/65797 |
| cosine | anchor | legacy | `8,1,2,2,1` | 44 | 100 | 1.25 | 5/8 | 1121/2102 | 1311/49 | 22092/64682 |
| cosine | tuned | legacy | `0,0.5,2,2,1` | 44 | 100 | 1.25 | 8/8 | 1353/2121 | 1334/26 | 23098/66403 |

All selected rows converged, had zero active-active continuation
infringements, passed the immediate deterministic repeat, and reproduced the
same quality counts on three rotated timing runs. The reference coverage and
active-piece coverage gates were both satisfied.

| Family | Solve seconds min/median/max | Wall seconds min/median/max |
| --- | --- | --- |
| none | 5.800 / 5.800 / 6.000 | 11.380 / 11.452 / 11.583 |
| linear | 7.700 / 8.000 / 8.200 | 13.270 / 13.588 / 13.631 |
| cosine | 6.000 / 6.000 / 6.000 | 11.488 / 11.546 / 11.629 |

Under the predeclared lexicographic reference objective, tuned `linear` is the
1024 winner: 8/8 exact references and 1369 correct constraints. Tuned `cosine`
is the coverage/consistency tradeoff: 16 fewer correct reference constraints,
but 21 more evaluated constraints, 5 more active pieces, 5 fewer Defects, and
a lower aggregate active-constraint infringement rate (34.78% versus 36.27%).
This is a 1024-only local search result. At the user's direction, the tuned
linear-normal/cosine-decision row and weights `0,2,2,2,1` were promoted to the
shared and CLI defaults before the deferred 2048 validation.

The reusable command surface was:

```bash
/tmp/vc_direction_ablation_runner.sh tune all
/tmp/vc_direction_ablation_runner.sh validate all
```

Complete tuple results and logs are under `/tmp/alignment-falloff-tuning`.
