# CT slice judge compute benchmark

Measured 2026-09-25 on the available RTX 5090 (32 GB). This benchmarks the
unchanged architecture proposed in `CT_SLICE_JUDGE_PLAN.md`; it is not a trained
judge or an end-to-end training benchmark.

## Results

Times include forward, loss, backward, gradient clipping, AdamW, and parameter
EMA. Three warmup updates precede ten measured updates, with CUDA synchronization
around each update. Compilation time is excluded. Effective follower batch is
eight, microbatch two. Judge-only uses eight full sequences; combined uses eight
follower states plus ten judge sequences, including the planned 25% synthetic
allocation. Branches execute sequentially with separate parameters/optimizers.

| Mode | Work per update | Mean ms | Median ms | p95 ms | Peak allocated GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| Eager | Follower × 8 | 50.8 | 50.8 | 51.4 | 0.39 |
| Eager | Judge × 8 | 280.8 | 280.8 | 284.2 | 8.23 |
| Eager | Follower × 8 + judge × 10 | 395.1 | 396.0 | 402.6 | 8.25 |
| Compiled | Follower × 8 | 31.6 | 32.0 | 32.3 | 0.30 |
| Compiled | Judge × 8 | 206.3 | 206.2 | 207.7 | 6.27 |
| Compiled | Follower × 8 + judge × 10 | 290.0 | 290.4 | 292.5 | 6.32 |

The compiled judge costs about 25.8 ms per sequence amortized over an eight-state
update. Combined compute is about 9.2 times follower-only compute. At this fixed
workload, 50,000 combined updates would consume about 4.0 hours of update compute,
versus 0.44 hours for follower-only. These are extrapolations, not measured full
training durations. The measured absolute cost does not by itself require
reducing the proposed model before testing its utility.

## Scope and limitations

- CUDA BF16 autocast with FP32 weights, channels-last convolution layouts, four
  CPU threads, seed zero, and the default disabled cuDNN benchmark setting.
- The actual `DirectFollower` and production geometry/confidence loss use
  synthetic GPU-resident crops, straight histories, and known synthetic targets.
- The standalone judge prototype uses 33 positions, three 257 × 257 views each,
  encoder widths 16/32/64 and strides 1/2/2, 8 × 8 full-view tokens, 5 × 5
  first-stage center tokens, and two width-128 decoder blocks with four heads.
  View encoding uses minibatches of 33, retaining gradients through every view.
- The nine existing preview positions are repeated to fill each 33-position
  sequence. Images contain CT, center marker, and support channels. Metadata and
  binary targets are compute fixtures; this does not test identity detection,
  masking correctness, real sequence construction, or learning quality.
- Inputs already reside on the GPU. Native CT reads, CPU sampling, transfer,
  data augmentation, collection, diagnostics, checkpointing, and extra seed or
  off-grid endpoint entries are excluded. No historic neural features are cached.
- Peak allocated memory is PyTorch tensor allocation for each case, including
  resident inputs/models/EMA/optimizer state; it excludes other GPU processes
  and is distinct from allocator reservation. Raw results also report reservation.
- The GPU had desktop processes but no separate model-training process at
  preflight. Ten iterations are a short compute measurement, not a capacity or
  sustained end-to-end throughput test.

## Reproduction

From `fiber_follow/`, using the existing environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../../.. \
  /home/sean/Documents/villa/vesuvius/.venv/bin/python \
  scripts/benchmark_ct_judge.py --out /tmp/ct-judge-bench-eager.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../../.. \
TORCHINDUCTOR_CACHE_DIR=/tmp/ct-judge-inductor \
TRITON_CACHE_DIR=/tmp/ct-judge-triton \
  /home/sean/Documents/villa/vesuvius/.venv/bin/python \
  scripts/benchmark_ct_judge.py --compile --out /tmp/ct-judge-bench-compiled.json
```

The [benchmark script](../scripts/benchmark_ct_judge.py) creates disposable models;
it does not load or modify the active training run or its weights. Raw results
with individual timings and software versions are saved locally in
`output/ct_judge_benchmark/eager.json` and `output/ct_judge_benchmark/compiled.json`.
Both runs completed with finite gradients checked during clipping.

## Encoder alternatives measured on 2026-09-25

The follow-up compares ways to retain center detail while reducing broad-context
compute. All cases use the same three views, 33-position history, 64 full-view
plus 25 center tokens, decoder, precision, widths, and training batch allocation.
These are benchmark options, not a change to the selected plan architecture.
Each compiled case has three warmups and 30 measured combined updates (eight
follower states and ten judge sequences). Only completed runs are reported; an
interrupted alternative run was restarted. Inputs remain GPU-resident.

| Encoder / center sampling | Mean ms | Median ms | p95 ms | Peak allocated GiB | Speedup vs fresh baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline / bilinear | 282.4 | 280.8 | 301.1 | 6.31 | 1.00× |
| Baseline / direct indexing | 284.1 | 281.8 | 297.7 | 5.53 | 0.99× |
| Pool native features before wider context / indexing | 189.3 | 188.5 | 198.1 | 4.29 | 1.49× |
| Native center crop + half-resolution context / indexing | 117.0 | 116.6 | 120.3 | 1.80 | 2.41× |

An eager CUDA operator profile identified convolution, normalization, and tensor
copies as substantial costs; attention kernels were comparatively small. Those
profile timings are diagnostic and are not substituted for the compiled update
measurements above. Raw operator records are in
`output/ct_judge_benchmark/profile_eager.json`.

**Direct center indexing:** The configured center points lie exactly on native
feature pixels. Index the 25 selected positions before converting to float32,
instead of converting the complete feature map for bilinear sampling. This
preserves the selected values and their gradients: CPU checks were bit-exact for
FP32 and BF16 maps of sizes 257 and 65. Results are recorded in
`output/ct_judge_benchmark/center_index_verification.json`. This optimization
saved memory, but did not produce a clear compiled speed improvement. General
nonintegral sample locations still require interpolation.

**`feature_pool2`:** Run the native first stage unchanged, extract its center
tokens, then apply 3 × 3 average pooling with stride two and padding one before
the two remaining context stages. This preserves the native center features at
fixed weights. Broad-context deep features change, with a 33 × 33 final map
instead of 65 × 65, still pooled to the same 8 × 8 token grid. This is the more
conservative representation change and cuts measured update time by 33%.

**`dual_scale`:** Apply the same first-stage weights separately to a native
65 × 65 center crop (eight trace voxels across) and the whole view after 3 × 3
average pooling with stride two and padding one (129 × 129). Only broad-context
features traverse the deeper stages. Center tokens keep the same physical
offsets and native input resolution. The context input spacing is 0.25 trace
voxels, still twice as fine as the follower's fine input; field of view remains
32 trace voxels. Odd pooling preserves the center lattice. Crop-local GroupNorm
statistics differ from the baseline, so identical native center pixels do not
imply identical encoded features. This cuts measured update time by 59%.

The two-scale variant is a promising compute/representation tradeoff. It retains
native evidence near the tracked point, broad neighboring context, temporal
sampling, and token count. However, it removes some fine neighborhood texture
before learning features. Neither faster variant has been trained or evaluated
on real departures. Preserving token count is not proof of preserving identity
information. The `feature_pool2` variant offers a less aggressive alternative.
Neither variant measures or reduces native source I/O in this prototype.

Reproduce the fresh baseline and variants from `fiber_follow/`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../../.. \
TORCHINDUCTOR_CACHE_DIR=/tmp/ct-judge-inductor \
TRITON_CACHE_DIR=/tmp/ct-judge-triton \
  /home/sean/Documents/villa/vesuvius/.venv/bin/python \
  scripts/benchmark_ct_judge.py --compile --cases combined --updates 30 \
  --out output/ct_judge_benchmark/baseline_repeat.json

for variant in baseline feature_pool2 dual_scale; do
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../../.. \
  TORCHINDUCTOR_CACHE_DIR=/tmp/ct-judge-inductor \
  TRITON_CACHE_DIR=/tmp/ct-judge-triton \
    /home/sean/Documents/villa/vesuvius/.venv/bin/python \
    scripts/benchmark_ct_judge.py --compile --cases combined --updates 30 \
    --center-sampler index --encoder-variant "$variant" \
    --out "output/ct_judge_benchmark/${variant}_index.json" || break
done
```
