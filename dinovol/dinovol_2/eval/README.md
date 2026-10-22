# Full-volume embeddings

## Color visualization of saved dense embeddings

Create a separate RGB OME-Zarr without rerunning inference or changing the embeddings:

```bash
.venv/bin/python -m dinovol_2.eval.embeddings_to_rgb \
  --input /mnt/raid_nvme/PHerc0332_L2_640k_embeddings.zarr \
  --array dense \
  --output /mnt/raid_nvme/PHerc0332_L2_640k_dense_rgb.zarr \
  --device cuda --threads 16 --writer-workers 16 --levels 6 --resume
```

This fits one global three-component PCA from a reproducible sample of nonzero
vectors distributed across the stored chunks. Components map to red, green and
blue with global 1st/99th percentile clipping. The saved PCA mapping stays fixed
across chunks, slices, pyramid levels and restarts. Similar colors indicate
similarity in that projection; colors are not class labels and three components
cannot show every difference in 64 dimensions. Zero vectors remain black.

The result is Zarr v2 / OME-NGFF 0.4 with C/Z/Y/X axes, three uint8 channels,
lossless `vc-delta3d` compression and six resolutions (full size through 1/32).
Each lower level is a 2x box mean of RGB, with zero padding at the volume edges.
With `--device cuda`, both color projection and pyramid averaging run on GPU.
The averaging is byte-identical to CPU: sums of eight uint8 samples are exact
in float32, and both paths round ties to even.
Coordinates remain in source-level voxel units; no physical voxel size is guessed.
OME channel metadata assigns additive red/green/blue display colors. A reader
must support the custom `vc-delta3d` codec (for example `vc_reader`); ordinary
Zarr/OME viewers without that codec cannot decode it merely from OME metadata.

Only stored input chunks are streamed, using bounded read/projection/write batches.
`--device cpu` also works. Source arrays must be complete, Z/Y/X/C, zero-filled,
have all channels in each chunk and use fixed-scale values (the dense array does).
`--resume` reuses the PCA mapping and committed progress, replaying unfinished
batches. Keep the source immutable; the resume fingerprint checks metadata and
chunk coordinates, not every source byte. The destination must be separate.

Regression coverage: `python -m unittest discover -s dinovol_2/tests -p test_embeddings_to_rgb.py`
checks sparse background, odd edges, channel-first codec roundtrips, fixed colors
across chunk boundaries, pyramid means, interruption/resume and source hashes.

## Generating embeddings

Run from the `dinovol` directory using its existing environment:

```bash
.venv/bin/python -m dinovol_2.eval.embed_volume \
  --checkpoint /mnt/raid_nvme/dinovol_ckpts/checkpoint_step_640000.pt \
  --volume /mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr \
  --level 2 --full-volume --compile \
  --dtype int8 --codec vc-delta3d \
  --output outputs/PHerc0332_L2.zarr
```

Omitting `--z-start` selects the entire volume; `--full-volume` makes that explicit.
The level-2 input has shape `(8398, 3941, 3941)` in Z/Y/X. All coordinates and
output grid steps are in voxels at the selected level. Use `--z-start Z --depth D`
for a region; the region depth defaults to 192 when only `--z-start` is supplied.

The output arrays span the requested extent. Processing streams through slabs of
two lattice strides by default (192 voxels for a 192-voxel tile with the default
32-voxel margin and ramp). `--slab-depth` changes that working depth and must be a
positive multiple of the lattice stride. The last slab is padded for inference
and clipped to the actual output extent. `--read column` reduces host memory use
by reading one column at a time instead of prefetching full-width bands.

The backbone and dense head are in evaluation mode with all parameter gradients
disabled. Every model call runs inside `torch.inference_mode()` (which also
disables autograd). Compilation is enabled by default: `torch.compile` uses
Inductor in default mode with static shapes for the actual `forward_features`,
`fpn_grid`, and `readout` methods. The first nonempty tile triggers compilation;
subsequent tiles reuse the compiled functions. Dense readout batches remain padded
to a fixed size. `--no-compile` selects eager execution for debugging.

For a local `vc_reader` cache, the script reads its recorded source URL and
infers the cache root. Cached chunks are reused; uncached chunks are fetched and
cached. Equivalently, pass the original `s3://.../volume.zarr` URL together with
`--cache-dir /mnt/raid_nvme/volume_cache`. `--no-cache-writes` still fetches misses,
but does not save them; `--offline` instead fails on uncached chunks. Ordinary
local Zarr inputs without cache metadata remain local inputs.

## Resume

Repeat the original command with `--resume`. The saved normalization statistics
are reused automatically, so a separate `--normalization-stats` file is not
needed when resuming.

Each slab is committed to the group's `.zattrs` only after all its chunk writes
finish. Committed slabs, including entirely masked slabs, are skipped. An
unfinished slab's chunks are removed and recomputed on resume. Individual chunk
files and progress metadata are replaced atomically. A process lock prevents two
CLI runs from writing the same output concurrently and releases automatically
when a process exits or is killed.

Resume checks the checkpoint SHA-256, source and level, volume shape, output
extent, lattice, slab plan, normalization, compilation settings, PyTorch version,
and array schemas. Keep the same
checkpoint path and computation/storage options. Read threads, cache location,
online/offline settings, band/column reads, writer workers, CPU thread count, and
overlap-buffer device may change. The input volume must
remain unchanged. Outputs made by older versions without a slab manifest cannot
be resumed. `--overwrite` starts fresh and removes the existing Zarr group.
Do not switch between compiled and eager execution within a resumed output:
compiler optimizations can change floating-point rounding.

## Throughput settings

For the PHerc0332 level-2 run on the 32 GB RTX 5090, retain the 192-voxel slab
depth, 192-voxel tiles, and 32768-point dense-head batch. Use these additional
options with the original command and `--resume`:

```text
--strip-device cuda --torch-threads 4 --threads 16 --writer-workers 16
--point-batch-size 32768 --read band --connections 32
```

`--strip-device cuda` avoids copying floating-point overlap strips to CPU and back
between columns. Consumed strips are released immediately, and the default
`--column-block-size 16` limits the retained row width. Reduce that block size
if another GPU workload leaves too little free VRAM.
Interpolation reuses device-resident shape constants instead of transferring them
for every point batch. These changes preserve the inference arithmetic, model
precision, quantization and chunk formats.

`--threads` controls vc_reader decompression; `--torch-threads` controls PyTorch
CPU operations; `--writer-workers` controls compression and filesystem writes.
There is one model process and one tile per backbone call. `--point-batch-size`
controls the separate dense-head point batch. Changing it can change quantized
results, so resume rejects a different value; older manifests imply 32768.

The [benchmark report](../perf/embedding/REPORT.md) records measurements and limits.
`python -m dinovol_2.eval.bench_embed_volume --help` describes the disposable-region
benchmark and profiler. It reads cached input only and never writes the production
embedding group. Network connection count was not tuned by the cached-input tests.

### Fused interpolation and asynchronous output

The default `--point-sampler auto` uses a forward-only Triton kernel on CUDA when
Triton is installed. It interpolates directly into the point feature buffer,
avoiding the eight-corner feature tensors and separate multiplication/reduction
passes at every feature level. The general sampler keeps its PyTorch autograd
path; CPU and installations without Triton also fall back to PyTorch.

The default `--async-output` stages quantized outputs into pinned host memory on
a separate CUDA stream. A bounded submission queue waits for copy events and
hands chunks to the existing compression workers while inference continues.
Slab commits still wait for every transfer and chunk write; failures propagate
before a slab is marked complete. Extra pinned host memory is bounded by four
staging blocks plus the existing bounded chunk queue.

The original full-volume command needs no additional options. These settings may
change on resume; `slab_runtime` records them for each newly committed slab.
Previously committed slabs retain their values. Interpolation allows small
floating-point rounding differences: the measured nine-cell region retained
byte-identical tokens, with roughly 5 per million dense and 10 per million ink
int8 values changing by one quantization step. Precision, model, resolution,
blending, quantization rules and lossless codecs remain the same.

Use `--point-sampler torch --no-async-output` to select the previous execution
path. The benchmark accepts the same switches for before/after measurements.
See the latest section of the [report](../perf/embedding/REPORT.md) for timings,
output comparisons and exact reproduction commands.

### Reusing tiles between slabs

Add `--slab-group-size 2` to the original command, retaining `--slab-depth 192`
and `--resume`. Two consecutive unfinished slabs are evaluated together, so their
shared boundary tile runs through the backbone once. The saved slab plan, chunk
layout, model precision and output extent remain unchanged. The default group
size is 1. Grouping increases working depth and overlap-buffer memory. Memory
also depends on the number of nonempty columns in each row; a short region's
peak does not establish capacity for the full volume.

`--column-block-size 16` (the default) bounds the number of output columns
processed per pass. Keep `--strip-device cuda`: each pass retains its row
overlaps on GPU, without transferring them to CPU. It rereads/recomputes one
left halo column per additional block, then writes disjoint output cells with
the same blending weights. With `--read band`, reads cover the block width,
not the full volume width. Progress restarts for each printed `column block`.
This bounds row-strip memory independently of volume width, at the cost of
some repeated halo computation. Width 8 further reduces memory; width 0 restores
the previous full-width traversal and its potentially large memory demand.
Block size can change on resume without changing output layout or precision.

The column loop also copies only the right overlap before freeing its
accumulator. If the CUDA transform's separate scratch allocator runs out of
memory, it releases unused PyTorch cache and retries once. This cache retry
alone cannot fix a row whose live strips consume too much VRAM; column blocks
bound those live allocations. For very small GPUs, reducing the slab group
size or using CPU strips remains possible.

A group never crosses a completed slab. All writes in the group finish before
one atomic metadata update marks its slabs complete; interruption retries the
entire unfinished group. `slab_runtime` records the grouped slab indices and
column block size. Work counts are recorded on the group's first slab, with empty completion records for
the others, so totals count each backbone invocation once. Group size may change
on resume. On the measured two-slab region, grouped BF16 output was byte-identical
to separate slabs.

### Optional FP8 transformer

`--transformer-precision fp8` enables lossy FP8 computation in the transformer
linear layers and attention. It requires `--compile`, Triton, and an NVIDIA GPU
with compute capability 8.9 or later; measurements were made on the RTX 5090.
The CNN, normalization, positional encoding, dense/ink heads, blending and output
quantization retain their existing precision. The default remains `bf16`.

The FP8 path uses dynamically scaled E4M3 inputs and weights, fast-accumulation
matrix multiplications, FP32 softmax statistics and attention accumulators, and
BF16 attention/linear outputs.
Compilation uses `max-autotune-no-cudagraphs` to select efficient FP8 matrix
multiplications. Initial compilation takes longer than a warmed-up run.
Currently supported checkpoints use full EVA attention, no separate fused QKV
biases, head dimensions at most 128, and linear dimensions divisible by 16.

Use a **new output group** for FP8. Starting from the original command, add:

```text
--transformer-precision fp8 --slab-group-size 2
--normalization-stats /mnt/raid_nvme/PHerc0332_L2_640k_embeddings.zarr/.zattrs
```

Change `--output` to a distinct path such as
`/mnt/raid_nvme/PHerc0332_L2_640k_embeddings_fp8.zarr` and omit `--resume` on
its first run. Subsequent FP8 runs can resume that group. Resume rejects a switch
between BF16 and FP8 so one volume does not silently mix the two computations.
Older manifests without a precision field imply BF16 and remain resumable.

FP8 differences exceed rounding noise. The [report](../perf/embedding/REPORT.md)
includes cosine similarity, worst-case differences, and throughput measurements.
These embedding comparisons do not establish downstream ink-detection accuracy.

## Compression and output size

`--dtype int8 --codec vc-delta3d` uses lossless vc-delta3d compression for every
embedding array, with all channels stored together in each chunk. Quantization
is unchanged: `dense` and `ink` use fixed-scale int8 (`value = q / 127`);
`tokens` and `tokens_unit` use int8 with a float16 scale per vector. Those scale
arrays use Zstd because the Delta3D writer accepts integer arrays. Read with
`read_embeddings()` to decode and dequantize. vc-delta3d is a custom codec and
requires a compatible reader such as the project's `vc_reader`.

Without `--codec`, dense/ink int8 outputs use vc-delta3d and token arrays use
Zstd, preserving the previous storage defaults. `--dtype float16` uses Zstd and
cannot be combined with `--codec vc-delta3d`.

Masked outputs remain zero and all-fill chunks are omitted. Storage still
depends strongly on requested outputs and resolution: 64 dense plus 32 ink
channels over this level-2 volume represent about 12.52 TB at stride 1 before
compression and sparse omission, excluding token arrays. `--dense-stride 2`
reduces those arrays to about 1.57 TB before compression; the stride must divide
the checkpoint patch size. `--outputs` selects which arrays to produce.

## Regression checks

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover \
  -s dinovol_2/tests -p test_embed_volume_cli.py -v
```

These CPU tests exercise tiled inference using a small synthetic backbone,
partial edges, streamed versus single-region results, interrupted and completed
resumes, asynchronous writer failures, configuration rejection, cache-source
resolution, locking, overwrite, and actual vc-delta3d encode/decode. The existing
`tests/test_embed_volume.py` also covers the real model and heads when CUDA is
available.
