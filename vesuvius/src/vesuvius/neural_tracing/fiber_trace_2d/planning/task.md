# Task: fix large staged Fiberlet cache preparation

`vc_fiberlets chunk-route-stats` succeeds for the established 512-base-voxel
region but fails while preparing a 1024-base-voxel staged region with:

```text
vc_fiberlets: scheduled fiberlet chunk did not resolve to data
```

Fix the cache identity/resolution bug without weakening validation or changing
the current generated anchor/Fiberlet data. An anchor cache written by an older
producer must never share a namespace with current Fiberlet generation when the
producer outputs differ. Preserve bounded memory, parallel preprocessing,
durable authoritative caches, and exact staged-reduction output.
