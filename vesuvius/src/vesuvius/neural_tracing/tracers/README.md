# Tracers

Neural tracers that follow a structure through CT crop by crop, and the parts they share.

| Package | Contents |
| --- | --- |
| `models/` | Crop models and their contract: the crop transformer (`regression`, `flow`), the sequence model, the registry and checkpoints |
| `shared/` | Geometry, observed-path references, the commit policy, path relocation, JSON/atomic output, batch tensors, the CT structure tensor |
| `shared/annotations/` | Fiber annotations: AFV collections, VC3D fiber JSON, the annotated-fiber type, kink repair, whole-fiber splits |
| `shared/crop_sampling/` | CT crop loading: local and remote chunked volumes, intensity normalization, oriented crop sampling, prefetch |
| `fibers/` | The fiber tracer: data, crop frames, training, tracing and evaluation (`fibers/README.md`) |
| `sheets/` | The sheet tracer: grows a surface row by row from a seed curve (`sheets/README.md`, `sheets/ARCHITECTURE.md`) |

Dependencies point one way: `models/` and `shared/` never import a tracer package such as `fibers/`, so another tracer
can reuse them unchanged.
