# Training span eligibility

The 3D file-backed loader filters training and test-loss data identically:
only `cspline` and `trace` spans are retained, regardless of length. Line points
outside the original first/last control points are removed. Rejected spans
split the fiber into contiguous components; no synthetic bridge is created.
Supervision uses the selected component only. Scroll-balanced sampling remains
uniform per scroll and uniform over retained arclength within the scroll.

NML explicitly annotated edges remain usable: their parser-generated legacy
Lasagna metadata does not describe actual interpolation. JSON without eligible
provenance is skipped; an empty dataset raises an error. Source files are not
modified. The separate 50-fiber tracing evaluator is deliberately unchanged.
Test losses and sample indices therefore differ from older training runs.

## New training

`train_5scroll_json_eligible_ema_16x8.json` uses the five-scroll train/test JSON
split, no NML or historical auxiliary test source, no embeddings, LR0.01,
128 patches/update (16 per GPU at full allocation), BF16/GroupNorm/stride4,
2M updates, EMA0.99/0.999, tests and checkpoints every100 updates.
It starts fresh; do not pass `--resume`.

```bash
source /home/hendrik/.venv_las/bin/activate
python -m vesuvius.neural_tracing.fiber_trace_3d.train /home/hendrik/villa/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_5scroll_json_eligible_ema_16x8.json
```

Manager handling remains automatic. After validation, the user-requested fresh
run was launched as `5scroll_json_eligible_128_sd1_16x8_ema_20260914_105436` in
`/ephemeral/hendrik/fiber3d_train2/runs`. It completed startup calibration and
entered adaptive training; manager scheduling may pause/resume it.

## Verification

Approved caller: `vesuvius/scripts/run_fiber3d_cooperative_tests`, selecting
`training-spans` in its adjacent JSON. CPU geometry, loader integration,
deterministic length-weighted sampling and actual corpus audit; no changed GPU
math and no GPU use required.

Validation:38 focused tests passed (2 CUDA skips),32.75s;317 existing sampling,
queue, loss and normal-loading regression tests passed (11 skips,2 expected
failures),26.06s. Approved runner output under
`/ephemeral/hendrik/fiber3d_train2/cooperative-tests/20260914_104925_629191`
and `20260914_105208_737537`. No training-speed claim is made for this data-policy
change. Ambiguous repeated CP coordinates fail explicitly; the full configured
JSON corpus passes the exact CP mapping.

| Scroll | Train sources → components | Test sources → components |
|---|---:|---:|
| PHerc0139 | 197 → 382 | 10 → 31 |
| PHerc0332 | 74 → 88 | 3 → 3 |
| PHerc1299 | 77 → 98 | 5 → 5 |
| PHerc1451 | 5 → 16 | 2 → 2 |
| PHercParis4 | 593 → 940 | 30 → 49 |

Splitting does not assign equal weight to each component. Original corpus paths
and detailed retained lengths are printed by the local regression test.
