# Thin Lasagna Port For Fiber 3D Inference

The current fiber 3D inference plan and partial implementation drifted too far
from Lasagna `preprocess_cos_omezarr.py predict3d`. Correct the plan before
implementation: fiber inference must be a minimal Lasagna-style port.

Requirement:

- Everything about tiled inference, output chunking, resume, temp cleanup,
  atomic writes, OME-Zarr group creation, scale-space pyramid generation, and
  `.lasagna.json` writing must be shared with or directly ported from Lasagna
  `predict3d`.
- Fiber-specific code may only specialize:
  - which model is loaded;
  - how raw model output channels are interpreted;
  - which persisted output channels are written;
  - how product completeness is defined for those persisted channels.
- Remove or collapse superfluous fiber-specific reimplementations that duplicate
  Lasagna behavior.
- Remove intermediate compatibility leftovers. Do not keep legacy aliases,
  duplicate output adapters, old raw-bundle constants, old manifests, or old
  tests/docs around merely so old fiber V0 imports keep working.
- Fiber inference output remains only:
  - `presence`, stored as uint8 fixed point where `0 == 0.0` and `255 == 1.0`;
  - `nx` and `ny`, stored with Lasagna's compact ambiguous hemisphere encoding.
- Fiber must not persist raw seven-channel option bundles.
- Fiber must not invent a custom primary manifest. The authoritative output is
  the `.lasagna.json` manifest.
- The fiber CLI should follow the Lasagna contract: output is a
  `.lasagna.json` path, not a directory containing a custom sidecar manifest.
- Fiber must not accumulate encoded `nx/ny`. Match Lasagna: accumulate raw
  model channels, then encode persisted channels only at product chunk
  finalization.
- Lasagna `predict3d` behavior and compatibility must remain unchanged.
