# Accessing Vesuvius Data

Volumes, segments and labels are read with [`vc_reader`](../../vc_reader/README.md); `vesuvius` has no reader of its own.
Remote scroll data is covered by the terms in the library README ([Data licence](../README.md#data-licence)).

## Finding published data

`vc_reader.load()` reads the Open Data catalog. `ScrollSource` browses one scroll or fragment, and common sample-name aliases resolve to it.

```python
from vc_reader import load, ScrollSource

catalog = load()
print([sample.id for sample in catalog.samples])

scroll = ScrollSource('PHercParis4', catalog=catalog)
print([volume.id for volume in scroll.volumes])
print([segment.id for segment in scroll.segments][:5])
```

See [discovery](../../vc_reader/docs/discovery.md) for segments, labels, surface volumes and their relationships.

## Opening a volume

```python
ct = scroll.volumes['20230205180739']
with ct.open(level=0) as reader:
    voxels = reader[100:104, 200:204, 300:304]  # Z, Y, X
```

Any Zarr group or array opens by path or URL, local, `http(s)://` or `s3://`. A group holds one array per resolution level.

```python
from vc_reader.reader import open_group, open_zarr

volume = open_group('/data/Scroll1.zarr')
print(volume.levels, volume['0'].shape, volume['0'].dtype, volume.voxel_size)
slice_zyx = volume['2'][20:200, 10:40, 50:90]

array = open_zarr('s3://vesuvius-challenge-open-data/PHerc0172/volumes/....zarr', level=1, cache_dir='/data/volume_cache')
```

`cache_dir` keeps remote chunks on disk. `memory_cache` gives the array an in-memory cache of that many bytes. Reads in DataLoader workers should use `threads=1`, since the workers are the parallelism.

## Normalization and tensors

Reads return values as stored. Normalize with `vc_reader.transforms`, whose statistics are per channel of the image it is given:

```python
import numpy as np
import torch
import vc_reader.transforms as A

zscore = A.Pipeline([A.Normalize('zscore', nonzero=False, eps=1e-8)], backend='cpu')
crop = zscore({'image': slice_zyx[np.newaxis].astype(np.float32)}, seed=0)['image'][0]
tensor = torch.from_numpy(crop)
```

| Mode | Result |
|---|---|
| `Normalize('minmax')` | `(x - min) / (max - min)` |
| `Normalize('zscore', nonzero=False, eps=1e-8)` | `(x - mean) / max(std, eps)` |
| `Normalize('percentile', lower=1, upper=99)` | clipped to the percentiles, then scaled to [0, 1] |
| `Normalize('fixed', center=mean, scale=std, bounds=(low, high))` | clipped to the bounds, then `(x - center) / scale` |

## Credentials

Public buckets need none. Private `s3://` locations take AWS keys from the environment or the shared AWS files (`~/.aws`); SSO, assumed roles and instance metadata are not used. HTTP basic auth passes `auth=(user, password)`, or `vc_reader.reader.load_basic_auth(path)` for a JSON file.
