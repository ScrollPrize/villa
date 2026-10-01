# Cached values at first trace divergence — 2026-10-01

Frozen raw u31500; original halfsigma_current_raw_v3 versus profile_chunks_u31500_v1/chunk32,chunk64,chunk128, under /ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3. Same reference cohort and model as reviewed_chunk_profile_20261001.md.

## Normalization and interpretation

Frozen model instantiated in eval mode:57 GroupNorm layers, zero BatchNorm, zero stored running-stat entries. There is no cross-sample normalization. Batch-shaped GPU kernels may change floating-point rounding, but that possibility is not isolated here. Inference uses bf16; overlap accumulation separately uses float16 with per-add rounding in arrival order. No precision setting changed.

At first pointwise divergence>0.001 base voxels in first trace segment, compare17³ output-voxel ROIs (34-base-voxel span), centered on original trace. Output scale2 verified from metadata. Presence differences below are uint8 code units, not percent intensity. Normal angle is unoriented axis angle, restricted to both presence>=32 (diagnostic filter, not an asserted tracer threshold).

|Output chunk|Case|Scroll|Restarts original→candidate|First index|Presence changed|Max presence Δ /255|Mean axis Δ|Max axis Δ|
|---|---|---|---|---|---|---|---|---|---|
|64|128|PHercParis4|1→2|247|1.93%|1|0.243°|6.46°|
|64|143|PHercParis4|1→0|14|0.61%|1|0.015°|1.43°|
|32|11|PHerc0139|1→0|4|2.40%|1|0.033°|2.03°|
|32|170|PHercParis4|0→1|19|1.82%|1|0.162°|6.76°|
|32|4|PHerc0139|0→1|45|4.68%|1|0.125°|2.88°|
|128|11|PHerc0139|1→0|49|0.75%|1|0.021°|2.07°|
|128|142|PHercParis4|0→1|1|0.48%|1|0.010°|1.22°|
|128|33|PHerc0139|1→0|211|0.00%|0|0.000°|0.00°|

The small presence deltas are real, including the same64³repeat. Some decoded axes differ several degrees; broad65³ROIs include rare much larger angular outliers (case11/chunk32 max83.74°, mean0.071°). Raw nx/ny bytes alone are misleading because hemisphere normalization can flip signs. These measurements do not isolate float16 versus bf16/kernel effects.

Case33/chunk128 is an important unresolved counterexample: traces are exactly equal through point210 then differ0.69809base voxels at211; all three channels are byte-identical across170495 common cached voxels in the65³ROI. That does not prove sampled beam inputs identical: missing margins, actual lookahead/candidate footprints, auxiliary normal inputs and search state are not captured. Do not attribute this case to accumulation or missing coverage from these measurements alone.

Some later ROIs are identical while trace separation exceeds8base voxels, consistent with previously different trace state persisting. No causal proof of chunk-seam artifacts. Full common coverage in6/8initial17³ROIs; case170/chunk32 has4624/4913 common voxels and case142/chunk1284369/4913. Missing margins excluded, not filled and compared as signal; absence there does not establish a missing tracer sample.

## Reproduction and validation

CPU-only, no inference or tracing rerun:
```
/home/hendrik/.venv_las/bin/python scripts/check_reviewed_divergence.py
/home/hendrik/.venv_las/bin/python scripts/check_reviewed_divergence.py --radius 32
/home/hendrik/.venv_las/bin/python -m pytest -q scripts/test_check_reviewed_divergence.py
```

Raw tables: reviewed_divergence_17_20261001.json and reviewed_divergence_65_20261001.json. Script asserts same case provenance, unique canonical source mapping, metadata scale/layout and in-bounds ROIs. Uses shared normal decoder and existing numcodecs codec on local format2 chunks. Synthetic cross-chunk decode and missing-mask test passed; independent diagnostic review incorporated.

Limitations: first-segment common prefix only, not complete beam execution replay; per-region results rather than exact cost inputs. Every changed-count comparison did have an initial-segment divergence. Zarr async open stalled in sandbox; stopped those read-only processes and decoded existing local chunks directly with numcodecs instead. No cache writes, GPU jobs or default changes. A fixed-tile fp16-vs-fp32 accumulation comparison and actual candidate-sample/cost capture remain follow-ups, not completed experiments.

