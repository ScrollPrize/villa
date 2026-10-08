# Corrected mini5 weighted4 fiberlet rerun

## Latest revision v3 — corrections000 and001

User additionally edited `evalextract_20260916T154925562_000001.json` in the same
corrected_fibers directory. SHA256:
67c8c5688b29631aa7a5cb4a5db134e17f583329bd93ebb0ac755c77f561f3ec.
Piece000 correction retained. Same algorithm, parameters, snapshot and caches.

Latest output: `/ephemeral/hendrik/fiber3d_train2/evaluations/mini5_corrected_weighted_i4_vis_20260916_v3`;
latest benchmark: adjacent `mini5_corrected_weighted_i4_vis_20260916_v3_inputs/manifest.json`.
All five rendered strips and vc3d exports completed; README.md indexes them.

| Metric | v2 (000 corrected) | v3 (000+001 corrected) |
|---|---:|---:|
| Reported restarts |5|5|
| Total eligible mm |15.5618|16.8873|
| Pooled mean mm |2.5936|2.8146|
| Median mm |1.0930|1.3195|
| P05 mm |0.1034|0.0843|
| P95 mm |3.8646|3.8646|

Every case still has one restart. Piece001 eligible length900→1452.5502base voxels
(2.1591→3.4847mm); failure at1400.2883base voxels,52.26vx before endpoint.
Other four replay objects exactly equal v2, including corrected000. Thus mean
change reflects increased reference extent, not a controlled algorithm improvement.
One run24.1466s including visualization; credited coverage100% (roundoff aside).
Independent plan review passed; source hashes/native eligibility and all5outputs
validated, git diff --check clean. No code changes or rebuild required this iteration.

Command:
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mini5_corrected_weighted_i4_vis_20260916_v3.json
```

## Previous revision v2 — correction000 only

The active corrected revision replaces piece000 with the user edit from
`/ephemeral/hendrik/fiber3d_train2/corrected_fibers/evalextract_20260916T154925562_000000.json`.
Raw SHA256: b7cd71fb6b444afc3228efbe40f700f1a45f7bb2bbab6175620e9bf07ee2261a.
Historical manifests/results and original annotations remain unchanged.

Output: `/ephemeral/hendrik/fiber3d_train2/evaluations/mini5_corrected_weighted_i4_vis_20260916_v2`.
New benchmark: adjacent `mini5_corrected_weighted_i4_vis_20260916_v2_inputs/manifest.json`.
This contains frozen corrected metadata/geometry, hashes, native eligibility plan,
prior annotation and corrected input provenance. Use this revision going forward.

Native CP/Lasagna eligibility excludes extrapolated tails. The corrected extent
is 1485.5264 base voxels versus900 previously, so this is NOT a matched-length
measure of a tracing algorithm improvement. Geometry and interpolation metadata
from the edited file are retained; only benchmark/source identity is restored.

| Piece | Scroll | Old length mm | New length mm | Old restarts | New restarts |
|---|---|---:|---:|---:|---:|
|000|0139|2.1591|3.5638|1|1|
|001|0139|2.1591|2.1591|1|1|
|019|1299|2.6389|2.6389|1|1|
|025|Paris4|4.3200|4.3200|1|1|
|026|Paris4|2.8800|2.8800|1|1|

Total length14.1571→15.5618mm; restarts5→5; pooled mean (distance/(restarts+1))
2.3595→2.5936mm. New median1.0930, P05 .1034, P95 3.8646mm.
Corrected piece failure arc1464.3755 of1485.5264base voxels, near the endpoint.
All four unchanged cases have byte-equivalent decoded replay JSON objects.
All five rendered images and vc3d reference/trace exports completed. Credited
reference coverage100%; all five failures distance_above_threshold.

Unchanged bestEMA999u22000 model67edd3a1…, weighted4 anchor fit, canonical128-base
fiberlet chunks, beam16, lookahead384, step48. Existing inference/construction
caches reused. RelWithDebInfo native binary. One run, wall28.0808seconds including
visualization; not a controlled performance comparison.

Command:
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mini5_corrected_weighted_i4_vis_20260916_v2.json
```

Optional corrections in benchmark-full are frozen into a fresh `_inputs` directory;
existing outputs are refused, not overwritten. Corrections currently require one
eligible run per input; split annotations fail explicitly instead of dropping runs.
