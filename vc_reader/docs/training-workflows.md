# Composable training workflows

For step-by-step reading, training and visualization, start with the
[notebooks](../examples/README.md). This guide covers the streaming integration
example and its lifecycle/resource controls.

[`scripts/training_workflows.py`](../scripts/training_workflows.py) connects
caller-selected crops to default PyTorch collation and CPU or CUDA augmentation.
The map-style Dataset lives in the example. It does not select patches, manage
splits or epochs, interpret masks, or choose supervision. Input recipes in
[`scripts/training_inputs.py`](../scripts/training_inputs.py) are shared with
[`benchmarks/bench_training.py`](../benchmarks/bench_training.py).

## Run an example

The existing environment needs the `reader`, `augment`, and `torch` optional
dependencies and native extensions. TIFF geometry additionally needs the
`labels` dependencies; independent benchmark interpolation needs SciPy from
`bench`. CUDA requires a CUDA-enabled build and device. No install command is
part of these examples. Catalog and ordinary NumPy reader imports do not load
PyTorch. The example module itself imports neither NumPy nor PyTorch until used.

From the repository root, with an existing build:

```bash
PYTHONPATH=src:benchmarks:scripts .venv/bin/python scripts/training_workflows.py custom \
  --native-dir build/cp314-cp314-linux_x86_64 --backend cpu --workers 2 --epochs 2

PYTHONPATH=src:benchmarks:scripts .venv/bin/python scripts/training_workflows.py paired \
  --metadata /home/sean/.cache/vc_reader/metadata-cff196b7c8a57b2d.bin \
  --native-dir /tmp/vc-reader-cuda --backend cuda --workers 2 --epochs 2 \
  --cache-dir /tmp/vc-reader-paired-cache --offline
```

Remove `--offline` to populate missing public arrays in the selected cache.
`--native-dir` is optional for an installed build. Every spawned worker loads
the same explicitly selected build before opening its first reader.

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

`render`, `project` and `published-transfer` use `/tmp/vc-reader-surface-cache`
in the recorded acceptance commands. They discover the published segment label
and its native geometry through the browsing API, then explicitly declare the
canvas pixel-to-UV convention `((1,0,.5),(0,1,.5))`. The volume affine and coverage
come from metadata. Only the pixel convention is caller-declared; this is not an
independent anatomical alignment measurement. Each recipe uses six 16×16 label
requests starting at `(18576+2*i,7984+2*i)`. Physical offsets, reference frames,
projection origins and bounds stay in sample metadata outside the batch.

```bash
PYTHONPATH=src:benchmarks:scripts .venv/bin/python scripts/training_workflows.py published-transfer \
  --metadata benchmarks/fixtures/training/catalog.json \
  --native-dir /tmp/vc-reader-cuda --backend cuda --workers 2 --epochs 2 \
  --cache-dir /tmp/vc-reader-surface-cache --batch-size 1
```

The checked-in catalog subset preserves the original sample's volume affine
records, both selected volume records and the segment record. Its provenance
file records the full snapshot hash. It also supports offline discovery tests
without a machine-specific metadata cache.

`published-target` pins the full catalog source ID ending in `/data/2` and
snapshot fingerprint `de38ba456eedfa4e0c4c6d6b4e093f4664d635eae551bfbe923bbdf6dc4ec58d`.
It samples a scalar prediction, not a categorical annotation. The CT-only
recipe retains both resolution candidates, verifies the selected header when
opening, and never resamples to satisfy the resolution query. The other 7.91 µm
candidate remains browseable and fails catalog/storage shape verification.

## Compose caller-owned requests

With `scripts` on `PYTHONPATH`, construct a Dataset using views already selected
through `ScrollSource` or the catalog:

```python
from training_workflows import (
    ReadSpec, TrainingDataset, image_arrays, label_arrays,
    pipeline, data_loader, restore_layers,
)
from vc_reader.torch import CudaLoader

# image is the render selected by label.surface_volumes.zarr(), as in the README.
arrays = {**image_arrays(layers=True),
          **label_arrays(masks=("supervision", "validation"))}
spec = ReadSpec(label, mode="paired", image=image, layers=slice(40, 65))
requests = [(0, (18576, 7984), (128, 128)), (0, (19344, 14192), (128, 128))]
dataset = TrainingDataset([spec], requests, arrays, seeds=[123, 456],
                          memory_cache_bytes=64 << 20, threads=1)
augmentation = pipeline(arrays, spatial_dims=2, backend="cuda")

# Run under if __name__ == "__main__" when using spawned workers.
with data_loader(dataset, workers=2, batch_size=2, pin_memory=True) as cpu:
    loader = CudaLoader(cpu, augmentation, seed_key="seed")
    iterator = iter(loader)
    try:
        for batch in iterator:
            image_bcdhw = restore_layers(batch)
            validity_bcdhw = restore_layers(batch, "image_valid")
            # Caller-owned model and supervision go here.
    finally:
        iterator.close()
```

For volumes use `ReadSpec(volume, targets={"label": label, "features": field},
level=...)`, ordinary CZYX arrays, and `spatial_dims=3`. For transfer select
`label.mappable_volumes[destination_id]`, inspect `required_operations`, then
use `mapping.volume` as the reference. A source requiring `surface_projection`
must use the geometry operation explicitly. The transfer example records
native and destination IDs, level, transform and known overlap. Missing
registration fails; nonoverlap produces false validity, which is distinguishable
from a valid all-background annotation.

Declare scalar, vector, normal and coordinate fields using `Array(...,
role="scalar"/"vector"/"normal"/"coordinate")`; geometric component order
is explicit (`components="xyz"` by default for 3D). Aligned sampling first
converts geometric values into the selected reference grid. The default
coordinate augmentation declaration treats those volume-anchored coordinates
as external values: geometry moves their samples without changing the absolute
coordinate values. Vector/normal components follow the augmentation geometry.
Read `sample.metadata["value_frames"]` before collation to inspect the output
component frame. A 3D component frame cannot be passed into a 2D pipeline.

## Layout, lifetime and budgets

Each request is `(reader_index, spatial_start, spatial_size)`. Dataset specs,
regions and uint64 seeds are serializable. Live readers, native caches and
cleanup callbacks are excluded from pickling. Readers open lazily and remain
reusable inside each process. Spawned persistent workers keep them across
epochs. The example closes workers at context exit, including early breaks and
exceptions; process finalizers close readers. Parent-opened data requires spawn.
A worker never constructs a CUDA reader or augmentation pipeline.

The shared decoded Zarr chunk budget is **per worker**, default 64 MiB across
all its opened readers. Reader thread count is per reader, default one.
Sampling tile and plan budgets are per reader (16 MiB and 4 MiB in this Dataset).
The native sampling work budget defaults to 64 MiB; encoded-file/decoded TIFF
caches and output buffers are separate. Multiply worker budgets by worker count;
PyTorch prefetch retains up to two CPU batches per worker and `CudaLoader`
prefetches one GPU batch. The CLI exposes chunk-cache MiB, reader threads,
PyTorch threads, workers and batch size. It defaults PyTorch CPU threads to one
to avoid thread-pool overhead on small collations. Embedding applications retain
control of their own PyTorch thread settings.

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

The example seeds are explicit and repeat across epochs and batch sizes.
For changing augmentation seeds, supply caller-selected occurrence seeds or use
CudaLoader's `seed` plus `set_epoch()` instead of `seed_key`. Sampler, distributed
rank and region choices remain caller responsibilities.

See [acceptance and measurements](training-workflow-benchmarks.md) for independent
comparisons, exact commands, measured overhead and outstanding real fixtures.
