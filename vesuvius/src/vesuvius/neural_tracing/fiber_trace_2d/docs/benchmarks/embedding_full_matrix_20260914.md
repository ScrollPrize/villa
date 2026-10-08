# Full embedding-policy evaluation — 2026-09-14

Status: running, not a completed comparison. Six evaluations of all50 locked
fibers (100 directions, six collections, five scrolls). No policy retuning.

## Policy and visualization

Source revision3b2f79701, native policy83e695caf. `anchors-lru`, capacity2,
observations512 base voxels apart, nearest touch, novelty distance0.10,
mean cosine cost, weight0.10, EMA mix0. No age expiry; each restart begins
with its seed anchor (capacity2 does not mean two initial anchors).
The64-restart result was the selected hard PHerc1299 fiber, not this full set.

Shared helpers: `volume-cartographer/core/include/vc/fiber_tracer/FiberEmbedding.hpp`,
`advanceFiberEmbeddingLru` and `fiberEmbeddingLruCost`.
Each run saves `full/cases/*/trace.json` plus `full/summary.json` with failure
positions/reference arcs; use the actual stored geometry for visualization.

## Snapshot matrix

| EMA99 snapshot | Updates | Samples | Test EMA99 loss | Variants |
|---|---:|---:|---:|---|
| Supplied tuning reference |6500|832000|1.056982199|Tuned, off|
| Best at capture |9900|1267200|1.013786527|Tuned, off|
| Current at capture |32000|4096000|1.221866208|Tuned, off|

Both updated snapshots were frozen before any evaluation from
`combined_5scroll_s1a_paired_contrastive_128_sd1_16x8_ema_20260913_174112`.
Best means smallest saved `test_ema_0p99/loss_total`, not best tracing score.
Full SHA hashes and config paths are in the sibling manifest and six configs.

## Reproduction and allocation

Use the existing approved entrypoint, with each manifest-listed config:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config CONFIG
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config CONFIG
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config CONFIG
```

Configs are single-use; new reruns require fresh output and inference namespace.
Eight GPUs, nonexclusive20GiB manager grant per GPU,96logical CPUs,batch1,tile256,
float32 accumulation, bf16 model inference, unchanged overlap96/border32.
Source caches remain; each variant has fresh prediction caches. Training may
share the GPU. Historical full evaluations used8GPUs/batch8/float16 accumulation;
runtime is not an isolated speed comparison and low-bit trajectory differences
are possible. The present setup matches the tuning inference precision/batch.

The initial single-GPU v1 attempt was interrupted at the user's request to use
all eight GPUs, after about4150tiles. Its partial artifacts remain untouched;
the final matrix uses fresh v2 outputs and does not count it as a result.

Own native RelWithDebInfo build in `/tmp/vc-live-native-cmake`, not the other
checkout's moving QuickBuild binary. Missing `<cstring>` include repaired;
no tracer algorithm changes. Binary/library hashes in the manifest.

## Reporting protocol

Classic-normalized equal-scroll score is the overall comparison, not a physical
mean distance. All per-scroll metrics: restarts,mean,median,P05,P95 and zero
segments. Complete reference partitions include terminal segments:
mean = reference length / (restarts + directed cases). Use shared
`fiber3d_comparison_report.py` and `replay_metrics.py`, not rounded old tables.
Historical rows include the September13 full finetune table and five older
collection snapshots; manifest records exact result roots.

The tuning fiber belongs to this evaluation set. Results are not an untouched
validation of a policy selected independently from all50 fibers. Small-scroll
support is limited; no statistical significance claim.

Read-only report command (fails until all six results are complete):

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/report_full_embedding_matrix.py
```

Use `--format json` for unrounded measurements/provenance; `--allow-partial`
explicitly labels pending rows and is only for monitoring. No source results
are modified. Validation:26 historical complete-cohort audits and three
report-control regression tests passed. All23 applicable embedding/runner
Python tests passed; one native diagnostic skipped due hardcoded build path.
Both actual native test suites passed in the isolated build.
