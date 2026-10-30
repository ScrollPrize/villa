# Qualzz AFV V2 volume inference

`python -m lasagna.qualzz_inference` adapts the published
[Qualzz20/afv_fiber_9um_v2](https://huggingface.co/Qualzz20/afv_fiber_9um_v2)
model to Lasagna's existing tiled inference engine. It reuses its incremental GPU
manager allocation, readers, probability blending, accumulators, parallel flush
and chunk-based resume. It does not change existing Fiber3D/Lasagna inference.

## Model and input

Provide a **reviewed local snapshot directory** containing `model.py`, `plans.json`
and `checkpoint_final.pth`. The adapter executes that local `model.py` and calls
its `load_model` and `normalize_ct` functions, rather than duplicating the model.
Do not use untrusted model Python files. The supported contract is V2's dictionary
output with `seg` logits ordered background, vertical, horizontal, intersection.
The upstream `dynamic-network-architectures` dependency must be installed in the
inference environment (optional Lasagna extra `qualzz`). Nothing is downloaded
or installed automatically.

Input is a local **uint8, ZYX Zarr array at 8.64–9.6 µm**, already downsampled.
For 2.4 µm Paris4, select level `2` (9.6 µm). This command does not fetch missing
CT chunks; populate/verify the input cache first. Keep input files immutable.

Default tiles are 128³ (the published training patch size), overlap 32, border
16. The upstream per-tile z-score normalization, including its standard-deviation
floor, is used on uint8-scale values. This is not global-volume normalization.
Inference and accumulation are FP32. Four-class softmax is applied before tile
blending; only V/H probabilities are retained. Auxiliary heads are not exported.

## Run

```bash
python -m lasagna.qualzz_inference \
  --model-dir "$MODEL" --input "$CT_LEVEL2" \
  --output "$OUTPUT/predictions.zarr" --voxel-size-um 9.6 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5
```

CUDA always uses the GPU manager. The usual `LASAGNA_GPU_MEMORY_GIB`,
`LASAGNA_GPU_PRIORITY`, `VILLA_GPU_MANAGER_SOCKET` and `CUDA_VISIBLE_DEVICES`
settings apply. Without a measured memory budget, workers request whole GPUs.
CPU execution is available with `--device cpu` for tests. Tile accumulators live
alongside the output store, not in `/tmp`. This is a foreground inference CLI,
not a new `las_manager` catalog model type.

Repeat the same command to resume completed chunk pairs. Checkpoint, model code,
plans, input path/shape and numerical settings are recorded and checked on resume.
Input chunk contents are not hashed; do not replace them during a run/resume.

## Standard output, no Lasagna manifest

The Zarr v2 group contains only two arrays plus descriptive root attributes:

| Array | Shape | Chunks | Meaning |
| --- | --- | --- | --- |
| `semantic` | `(2,Z,Y,X)` | `(2,128,128,128)` | V/H probabilities at input resolution |
| `tracing` | `(2,Z//2,Y//2,X//2)` | `(2,64,64,64)` | 2× mean-pooled V/H probabilities |

Both are uint8 (0–255), Blosc/Zstd compressed. Chunk size is configurable with
`--chunk-size` (even). All channels share each spatial chunk. There is no
`lasagna` attribute, sidecar manifest, or additional pyramid. The tracing array
is pooled from **blended floating-point probabilities before quantization**;
odd final planes are omitted, matching the generator. Explicit compressed zero
chunks record completed empty predictions; untouched empty input tiles can remain
absent. Paired publication ensures an interrupted pair is recomputed on resume.

Feed the result directly to the unmodified updated generator:

```bash
avf-spline-generator \
  --predictions "$OUTPUT/predictions.zarr" --array semantic \
  --tracing-predictions "$OUTPUT/predictions.zarr" --tracing-array tracing \
  --voxel-size-um 9.6 --storage compressed --field-cache-mib 16384 \
  --output "$OUTPUT/fibers.afv" --work-dir "$OUTPUT/tracing-work"
```

The generator may use a separate environment: its compressed backend requires
Zarr 2.x, whereas Lasagna supports Zarr 3.x writing v2 stores. Extraction defaults
to CPU; GPU extraction must separately reserve resources through the manager.

## Validation

CPU tests: `python -m pytest -q lasagna/tests/test_qualzz_inference.py`.
They cover four-class softmax, upstream normalization dispatch, pooled-before-
quantized output, odd edges, v2 layout, paired completion/provenance validation,
and serial/spawned-flush inference with exact expected output and resume.
These synthetic tests do not establish full-checkpoint throughput or GPU memory.
