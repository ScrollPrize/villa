# u31200 VC3D / Lasagna cache export

## Related TensorBoard test image

Original `test_sample_3d/principal_slices` at3980800samples (display3.981M) extracted to `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/halfsigma_current_raw_v2/test_images_3980800/test_principal_slices_samples_3980800_00.png`.2642×656PNG,812969bytes, no transformation. Adjacent manifest.json records sourceevent/tag/walltime/hash. Exactly1event found, PILverified and readbackSHA checked. Reproduce with approved runner action`extract_reviewed_test_image`. This training-image step differs from evaluated rawu31200 (3993600samples).

## Cache export

Directory to copy:

`/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/cache_sync_u31200_v1/halfsigma_current_raw_v2/vc3d_channels/`

Each normalized scroll subdirectory contains `inference.lasagna.json`, relative presence/nx/ny Zarr references and `inference.json` provenance. Open the Lasagna manifest for that scroll in VC3D. Output lattice is level1 (2base voxels),64³chunks. Sparse prediction coverage from the completed91fiber/182direction evaluation, not a full-scroll inference.

| Scroll | Complete spatial chunks | Transfer GiB |
|---|---:|---:|
| PHerc0139 | 962 | 0.526 |
| PHerc0332 | 30 | 0.015 |
| PHerc1299 | 495 | 0.269 |
| PHerc1451 | 32 | 0.016 |
| PHercParis4 | 2937 | 1.580 |
| Total | 4456 | 2.407 |

Exact channel-export payload including metadata:2,584,204,574bytes.13,368chunk symlinks, all resolved and validated. No incomplete channel triplets, scratch directories, mmap files or checkpoints exported. This inventory contains all published coverage in this model-specific cache; no new inference/GPU work was needed. Old exports unchanged. Source SHA256 for all5scrolls matches rawu31200:3869682f2bb31942dbc731b7b7816dd8207e138d013b215cf5aed1f8c8860f1f.

Copy with symlink dereferencing:

```bash
rsync -aL --info=progress2 nat64:/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/cache_sync_u31200_v1/halfsigma_current_raw_v2/vc3d_channels/ ./u31200_predictions/
```

Do not copy the parent evaluation directory, which also includes models and unrelated artifacts. Local export depends on its original cache links; `-L` produces a standalone copied dataset. Manifest paths remain relative and portable after copying.

Reproduce: set scripts/training_failure_mining_control.json to `{"action":"reviewed_cache_export","cache_export_model":"halfsigma_current_raw_v2"}`, run `bash /home/hendrik/villa3/scripts/run_training_failure_mining`. Export inventory is immutable; if cache coverage changes use a new export version. Two regression tests passed; independent review approved; verified normalized labels, all manifests/array metadata, model hashes, link counts and scratch exclusion. Machine-readable sizes/inventory:parent `export_report.json` and per-model `inventories/`.
