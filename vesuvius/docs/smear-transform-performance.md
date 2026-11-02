# SmearTransform: compiled kernel

`SmearTransform` (used by the default ink training augmentation and by the generic
training pipelines) blends every slice with the mean of the `k` slices before it,
**after** those were themselves updated. That recurrence cannot be vectorised over the
smear axis, so the torch implementation runs a Python loop of about 25 tiny ops per slice
and spends its time in dispatch overhead. For the default ink patch `(1, 17, 128, 128)`
with `smear_axis=3, num_prev_slices=3` it was the single most expensive transform of the
default preset (about 41-47 % of the preset's CPU time per sample).

`float32` CPU images now run the same recurrence in a small numba kernel
(`noise/_smear_kernel.py`). numba is already a declared dependency. Other dtypes, GPU
tensors, or an environment without a usable numba keep the previous torch implementation.

## What did not change

The kernel uses the same float32 operation order as the torch code (previous slices are
summed from +0.0 in order, rolled, divided by `k`, then blended with `(1 - alpha)` and
`alpha`), so results are bit-for-bit identical:

* `tests/models/augmentation/test_smear_transform.py` compares the compiled path with the
  torch path bit-for-bit for eight shape/axis/shift/`k` combinations (including 1-D slices,
  four spatial dimensions, negative shifts and axes no longer than `k`), the torch fallback
  for `float64`, and the fallback when the kernel module cannot be imported.
* Running the whole default augmentation preset on 120 seeded samples (all of `image`,
  `inklabels`, `supervision_mask`) produced the same SHA-256 before and after the change.

One deliberate difference: the torch implementation wrote its result back into the
input tensor as a side effect (it blended through a view of the input). The compiled path
returns a new tensor and leaves the input untouched; the pipelines already replace the
input with the returned value.

## Measured effect

Single thread, otherwise idle CPU, torch 2.12.1 (CPU), numba 0.68.1, Python 3.14,
`(1, 17, 128, 128)` float32, `shift=(5, 0), alpha=0.2, num_prev_slices=3, smear_axis=3`:

| | torch implementation | compiled kernel |
| --- | ---: | ---: |
| One `SmearTransform` call (median of 22 interleaved runs) | 13.8-14.0 ms | 0.98 ms |
| Default augmentation preset, per sample, 300-400 samples (two interleaved rounds) | 9.4-9.7 ms | 6.1-6.3 ms |
| Share of that preset taken by smear | 41 % | 10 % |

The smear is applied with probability 0.3, so the average per-sample saving is
about 3.3 ms of a 9.6 ms preset (-35 %). On a busier shared host the same preset measured
16.9 ms per sample with smear at 46.7 %.

The kernel is JIT-compiled once per process (about 0.55 s on first use, cached on disk by
numba afterwards).

## Limits

This measures the augmentation step only. It does not measure end-to-end training: zarr
reads, decompression, collation and the GPU step are not included, so the effect on
`samples/s` depends on whether a given run is limited by augmentation CPU time. It changes
no model, label, loss or sampling behaviour.
