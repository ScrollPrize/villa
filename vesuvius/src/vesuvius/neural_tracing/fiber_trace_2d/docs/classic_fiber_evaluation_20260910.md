# Classic Fiber checkpoint evaluation — 2026-09-10

Status: **real-model pilot passed; both source-binding decisions confirmed, full100-direction cohort launched on GPUs0–7. Completion pending.**

Results/config/audit root: `/ephemeral/hendrik/fiber3d_train2/evaluations/classic_s1_91500_20260910`.
Native binary: `/tmp/vc-live-native-cmake/bin/vc_fiber_trace_metric`, current-source GCC13 RelWithDebInfo build.

## Dataset and model

Locked test partition: `fiber3d_train/2026-09-09_x4_collection/split_95_5_length_seed20260910/test`.
All50 file names and SHA256 values were checked against the original split manifest.
This is a newly locked evaluation partition, **not proven unseen by the classic model**.

| Collection | Scroll / scan | Test fibers | Binding status |
|---|---|---:|---|
| PHerc0139 | PHerc0139 / 20260102150214 | 10 | Catalogue CT + normals, annotation metadata and bounds pass |
| PHerc0332 | PHerc0332 / 20251211183505 | 3 | Catalogue CT + normals, prior annotation manifests and bounds pass |
| PHerc1229 | PHerc1299 / 20260309130042 | 5 | All files identify1299; user confirmed this mapping; bounds pass |
| PHerc1451 | PHerc1451 / 20260319101107 | 2 | Catalogue CT + normals, annotation metadata and bounds pass |
| PHercParis4_eval_fibers | PHercParis4 / 20260411134726 | 24 | User-approved completed local normals; source/bounds pass |
| PHercParis4_fibers | PHercParis4 / 20260411134726 | 6 | Same scan/cache as previous row; separate metrics |

Classic multi-scroll checkpoint: `/home/hendrik/runs/s1a_128_1_single_8x8_20260801_084232/snapshots/best91_5k.pt`.
SHA256: `f389da7914a6da34506f92204bf5441964e96599339dfe79dfc9c48b67165e17`.
This matches saved inference provenance for all five prior L1 multi-scroll runs.
The `best_140_7k.pt` snapshot is instead the previous Paris4 L0 run.

Fresh catalogue URL: `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json`.
Downloaded metadata is gzip-compressed; preparation hashes its decoded content and retains digest in the audit.
The older cached catalogue listed Paris4 normals, but the fresh document does not. Do not silently fall back to stale metadata.
User-approved explicit replacement is completed local Lasagna run `PHercParis4-20260411134726-las-sd2-5b17ff6c`, verified source PHercParis4/20260411134726. Full preparation validates this through the shared manager completed-dependency resolver and records the manifest hash.

The1229 label comes from the original supplied source prefix `s3://philodemos/david/PHerc1229/1229_fibers/`, recorded in the collection SOURCES.md. It appears to be a directory-name typo, not a coordinate transform or alternative scan.

## Pilot result

Whole PHerc0139 fiber `lt_20260909T071741261_000094.json`, not cropped or shortened.
Length3054.87736 base voxels =7.32865mm; both directions14.65730mm.

| Direction | Complete | Failures |
|---|---|---:|
| Forward | yes | 0 |
| Reverse | yes | 2 |

- Cold run wall time143.65s, GPU0 only;18 model tiles,26 coherently published output chunks (78 scalar-channel files).
- Worker-reported inference time18.19s, input loading36.27s. These are counters, not a complete disjoint wall-time breakdown; native tracing/coordination/metadata/accumulation also consume time. No throughput scaling claim from one pilot.
- Shared slots320MiB. InputL1 (2x base spacing), outputL3 (8x base); model tile512, overlap96, border32, bf16 inference/f16 accumulation, output chunk32.
- Same native VC3D beam8/lookahead2, nested threads1, step4 trace voxels (8base voxels), ellipse normal threshold20base voxels with existing tangential multiplier/reset behavior.
- Both native and coordinator summaries exist, no incomplete markers. All26 ready chunk fingerprints/lengths match and all78 channel payloads decode at expected size. Presence range0–245, nx1–255, ny3–251; trace JSON contains no nonfinite numeric constants.
- Ledger also retains998 partial, **unrequested** neighboring chunks touched by18 tiles; they are not marked ready or treated as complete inference. Only requested/ref-prefetch completion defines this sparse evaluation.
- Rendering off; segmented trace/reference JSON+OBJ available in `pilot/cases/`.

## Reusable command

No activation or installation required:

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare-pilot
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-pilot
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py pilot
```

The pilot already exists; launching it again deliberately refuses to overwrite results. `prepare-pilot` and `validate-pilot` are idempotent metadata/integrity checks. Full preparation passed and the following command launched after both user confirmations:

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --confirm-pherc1229-is-pherc1299
```

Output is `full/`; full.json and full_bindings.json persist settings and audit. Do not launch a second writer. All eight H100s were idle at launch; /dev/shm~1008GiB and /ephemeral~8.2TiB free at setup.

## Fixes and validation

Remote real-data preflight exposed native `Volume.root_attrs` as local-only. Live runner now fetches remote `.zattrs` via the canonical native HTTP URL, with voxel reads unchanged on the native fast path. Synthetic replay harness passes62 tracer cases +1 cache case +25 Python tests (11.86s). New locked-membership regression passes1 test. Preparation/launch bind checkpoint, original annotation membership/hashes, effective config and normal manifest hashes.

No full-cohort quality result, eight-GPU performance result, GUI integration, training hook or stuck-input cancellation claim. No training or original data changed. No commit requested.
