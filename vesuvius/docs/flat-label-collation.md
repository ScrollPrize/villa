# Project flat labels before pinning and accelerator transfer

Flat training already reduces ink labels and supervision along Z before forming
its 2D target and ignore mask. The loader previously pinned and transferred the
complete two CZYX label volumes first. Flat collation now takes their depth
maxima after per-sample augmentation and before stacking/pinning. The existing
loss projection over the retained singleton Z axis returns the same result.

Images and auxiliary fields retain their shapes and values. Dataset samples
still expose the complete CZYX volumes. Native/full-3D loaders use PyTorch's
normal collation. Custom hooks that inspect a *flat batch* should expect labels
with Z=1; dataset-level and native batch interfaces are unchanged.

## Controlled real-data measurement

Four adjacent public Paris4 `w00_20231016151002` level-2 crops at Y chunk 10,
X chunks 37–40, Z=24:41, 128x128 pixels. The original 17-depth label and
supervision voxels were used, not synthetic repeated labels. Their depth maxima
match the source middle-plane labels for these crops.

The harness repeats these four crops with batch 64, two spawned workers,
prefetch factor 2, pinned memory, and the same AMP fp16 setup in both modes.
It uses the upstream local-3D stem and Encoder/Decoder in a specified four-stage
U-Net (3,564,580 parameters), fixed divide-by-255 input normalization, SGD,
clipping, initialization and sampling order. The baseline is villa 37d1897.
It includes worker fetch, transfer, target projection, forward, loss, backward,
clipping and optimizer updates. It excludes Zarr I/O and augmentation, and is
not a run of the complete official trainer or its autoconfigured model.
The existing test environment uses PyTorch 2.7.0a0, older than the declared
project requirement >=2.12. These are exploratory component-pipeline results.

Six alternating fresh-process pairs; ten warmup updates, twenty measured
updates, first pair excluded. Every measured update completed. Peak GPU numbers
are allocated bytes; host PSS is a warm snapshot summed over parent and workers,
not peak host RAM. Per-step finite checks and synchronization serialize latency.

| Measurement | Original | Project labels |
| --- | ---: | ---: |
| Total image + two-label batch payload | 204 MiB | 76 MiB |
| Median warm parent + worker PSS | 4,695,993 KiB | 3,430,886 KiB |
| Median peak CUDA allocated | 5,607,223,296 B | 5,473,660,928 B |
| Main median, twenty complete measured steps | 3.001 s | 2.634 s |
| Independent pair using final helper, twenty steps | 2.730 s | 2.637 s |

Payload fell 62.75%; the host PSS snapshot fell 26.94%; peak GPU allocated fell
2.38%. The independent pair reproduced these memory measurements. Runtime fell
12.24% in the main series but only 3.42% in the independent pair: this does not
establish a stable 12% training speedup. Both measurements produced identical
model state, optimizer momentum, logits, scalar loss and metric dictionaries.
Four repeated crops and a short comparison establish no prediction-quality gain,
long-run convergence, whole-trainer throughput, distributed result or prize outcome.

Regression checks: `python -m unittest discover -s vesuvius/tests/ink_detection -p test_flat_collation.py` in an already prepared environment. Checks cover flat
projection, post-geometry NaNs/ignored labels, input and auxiliary immutability,
and native loaders retaining their complete Z depth. Full trainer integration
and supported-version performance still need verification.

Data provenance/terms: https://scrollprize.org/data (CC-BY-NC 4.0 unless noted).
