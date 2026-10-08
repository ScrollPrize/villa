# EMA99u5200 visualization

Completed all50fibers/90eligible runs/180directions:180inspection JPEG pages,
32305aligned cross-sections. Paired CT/presence reference and traced top/side
views, restart IDs and arc metadata; corrected per-fiber combined mean statistics.

Output root: `/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema99_u5200_vis_20260914_v1`

- `README.md`: full gallery index, statistics, most restarts.
- `per_fiber.csv` and `per_fiber.json`: all50fiber statistics and case mapping.
- `full/cases/<CASE>/rendered/strips_0.jpg`: inspection page.
- `full/cases/<CASE>/rendered/render.json`: section positions, restart IDs/arcs, geometry.
- [Highest restart-rate fibers and case IDs](eligible_ema99_u5200_worst_fibers_20260914.md).

Pinned snapshotSHA ccf706ab68d51c76910a8134b6970c3289a1e2503320d74c8eb75fa1f74209a6,
update5200/samples665600. Normal40/tangent80/grace160,episode80,no embedding.
All180 replay JSON payloads and aggregate metrics equal prior nonvisualu5200:
148reported restarts,46.091458mm combined distance/(restarts+1).

Shared renderer now uses actual replay factor for transverse frame, descriptor
and margins; defaultfactor4 preserved. Invalid factors rejected; aligned/oblique
and invalid-normal geometry covered by regression tests.22native+138Python
passed; all180page files/metadata audited. No numeric changes to tracer.

Wall1762.651162s (~29m23s), all8GPUs exclusive managerpriority100. Reused9150
model inference chunks;36543additional tiles/9293publishedchunks for the larger
off-trace inspection area. Not a cold-inference speed benchmark. CT/normal inputs
and model-keyed namespace reused; no cache key changes.

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_eligible_ema99_u5200_vis_20260914_v1.json
```

The existing output is complete; retain it and choose a fresh output directory
for any future rerun. The inference namespace may remain unchanged for this model.
