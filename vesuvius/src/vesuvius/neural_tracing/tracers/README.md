# Tracers

Neural tracers that follow a structure through CT crop by crop, and the parts they share.

| Package | Contents |
| --- | --- |
| `models/` | Crop models and their contract: the crop transformer (`regression`, `flow`), the sequence model, the shared blocks (`blocks.py`: the crop CNN with its feature-map, per-cell-token and per-crop-token heads, the two-layer `MLP`, transformer layers), the registry, checkpoints and weight loading (renames of earlier weight layouts) |
| `shared/` | Geometry, observed-path references, the commit policy, path relocation, JSON/atomic output, batch tensors, the CT structure tensor, training-run pieces (run dirs, log, LR schedule, EMA), compile options |
| `shared/annotations/` | Fiber annotations: fiber stores (VC object store), VC3D fiber JSON, the annotated-fiber type, kink repair, whole-fiber splits |
| `shared/crop_sampling/` | CT crops: the volume spec, intensity normalization, oriented crops (`CropSpec`) read with vc_reader (CPU, or its CUDA reader), GPU sampling of in-step crops from device blocks |
| `fibers/` | The fiber tracer: data, crop frames, training, tracing and evaluation (`fibers/README.md`) |
| `sheets/` | The sheet tracer: grows a surface row by row from a seed curve (`sheets/README.md`, `sheets/ARCHITECTURE.md`) |

Dependencies point one way: `models/` and `shared/` never import a tracer package such as `fibers/`, so another tracer
can reuse them unchanged.
