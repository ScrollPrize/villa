# Training integration benchmark

[`bench_training.py`](bench_training.py) measures reader overhead, array selection,
default PyTorch collation, host pinning, upload, CPU/CUDA augmentation, and
persistent-worker epochs. It uses six fixed requests, two warmup passes, and ten
measured passes. Independent readers and reference implementations validate the
samples before timing.

[`training_support.py`](training_support.py) owns the benchmark Dataset, array
layout declarations, augmentation setup, and worker cleanup.
[`training_inputs.py`](training_inputs.py) owns its synthetic fixtures and pinned
published selections. These modules are internal benchmark support, not a
library training API. Downstream applications own datasets, sampling, splits,
supervision, and training loops. For runnable examples, see the
[notebooks](../examples/README.md).

## Run

Use an existing build and environment with the `reader`, `augment`, and `torch`
optional dependencies. TIFF geometry also needs `labels`; independent benchmark
interpolation needs SciPy from `bench`. CUDA measurements require a CUDA build
and device. From the repository root:

```bash
AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks .venv/bin/python benchmarks/bench_training.py custom \
  --native-dir build/cp314-cp314-linux_x86_64 --workers 2 \
  --output /tmp/training-custom.json

AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks .venv/bin/python benchmarks/bench_training.py published-transfer \
  --metadata benchmarks/fixtures/training/catalog.json \
  --native-dir /tmp/vc-reader-cuda --cuda --workers 2 --batch-size 1 \
  --cache-dir /tmp/vc-reader-surface-cache --output /tmp/training-transfer.json
```

Use `--offline` after populating the remote-data cache. Each spawned worker loads
`--native-dir` before opening readers. `--cuda` adds GPU stages to the CPU
measurements. `--torch-threads` defaults to one.

## Workloads

| Workflow argument | Explicit input and output |
| --- | --- |
| `paired` | PHerc0009B, volume `20250820154339`, segment `20250919125754`, ink set `20260918`; six published 128×128 patches, layers 40:65, supervision and validation |
| `custom` | Local two-channel, three-layer image and categorical label attached through `with_source(kind="image")` and `same_grid=True` |
| `ct` | PHercParis4 at 8 µm ±2%; explicitly selected volume `20230206171837`, stored level 0; six 4³ crops starting at `(5264+i,3906,4158)` |
| `registered` | Local CT, categorical label, quality mask and vector field with explicit volume registration and a rotated value frame |
| `published-target` | PHercParis4 volume `20260411134726`, CT level 2, published surface model `20260413222639` stored level 0; six 8³ crops at `(9000+2*i,3600,3600)` |
| `published-transfer` | Published PHerc0009B ink projected from volume `20250820154339` into `20250521125136` using its catalog affine; B CT, resampled A CT, labels, validity, supervision, validation and conflicts; **use `--batch-size 1`** |
| `transfer` | Local label and named `scan_a` image transferred from A to B through an explicit `x+2` transform, with coverage retained |
| `render` | Published PHerc0009B geometry and ink, with explicitly registered label-pixel-to-UV mapping and offsets `(-9.6,0,9.6)` µm |
| `project` | Same real geometry and label, projected at thickness 9.6 µm; **use `--batch-size 1`** |
| `local-render`, `local-project` | Analytic planar geometry with independent known CT/label expectations |


Published surface workflows declare the label-pixel-to-UV convention
`((1,0,.5),(0,1,.5))`; the volume affine comes from the catalog. The checked-in
catalog subset preserves the selected volumes, segment, and affine records;
its provenance is in `fixtures/training/provenance.json`. Each report records
source selections, requests, transforms, coverage, and independent comparisons.

## Layout, lifetime and budgets

Each request is `(reader_index, spatial_start, spatial_size)`. Dataset specs,
regions and uint64 seeds are serializable. Live readers, native caches and
cleanup callbacks are excluded from pickling. Readers open lazily and remain
reusable inside each process. Spawned persistent workers keep them across
epochs. The benchmark closes workers at context exit, including early breaks and
exceptions; process finalizers close readers. Parent-opened data requires spawn.
A worker never constructs a CUDA reader or augmentation pipeline.

The shared decoded Zarr chunk budget is **per worker**, default 64 MiB across
all its opened readers. Reader thread count is per reader, default one.
Sampling tile and plan budgets are per reader (16 MiB and 4 MiB in this Dataset).
The native sampling work budget defaults to 64 MiB; encoded-file/decoded TIFF
caches and output buffers are separate. Multiply worker budgets by worker count;
PyTorch prefetch retains up to two CPU batches per worker and `CudaLoader`
prefetches one GPU batch. The benchmark fixes these cache budgets and uses one
reader thread. Its CLI exposes PyTorch threads, workers and batch size. It
defaults PyTorch CPU threads to one to avoid thread-pool overhead on small
collations. Embedding applications retain control of their own PyTorch thread
settings.

CDHW image arrays become `(C*D,H,W)` for an explicitly 2D pipeline. Per-layer
validity undergoes the same YX transform; it is broadcast across image channels
when needed, without reducing depth. Scalar channel/layer metadata restores
BCDHW after default collation. Labels and companion masks stay CHW. Projection
returns CZYX; geometry-dependent output sizes require compatible batch groups,
batch size one, or caller-specified padding. No hidden padding policy is applied.

`A.Image` receives normalization/intensity transforms. `A.Label` preserves
categorical label and companion-mask values, including 128 and 255; only
geometric validity uses boolean `A.ValidMask`. `uint8` continuous data stays
compact through CPU reading/collation and is promoted on the GPU by CudaLoader.
Other continuous types are explicitly converted to float32; CPU augmentation
promotes uint8 in the training process. Reads remain valid across subsequent
reads and reader closure, and prefetched CUDA outputs remain owned by PyTorch.

The benchmark seeds are explicit and repeat across epochs and batch sizes.
This makes repeated benchmark epochs comparable.

See [acceptance and measurements](../docs/training-workflow-benchmarks.md)
for recorded results and real-data reproduction commands.
