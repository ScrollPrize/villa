# Plan

1. Share the existing central-chord tangent and minimal frame transport between construction and corrections, rather than defining another tangent.
2. Interpolate CP target axes in a common transported frame; unset endpoints target the sampled axis. Spans with no manual endpoints retain sampled normals.
3. Inject the resulting normal field before ordinary frame alignment, resampling and smoothing. Delete all post-construction rotations and separate sign handling.
4. Test short-span injected-normal equivalence, equal CP normals over a 170-degree baseline, shared tangent behavior and existing annotation tests; rebuild VC3D.

## Spec Update

Corrected normals are ordinary display input normals. One regular tangent definition; no post-frame rotations. Common transported frame for target interpolation.

## Docs Updates

Update fiber annotation documentation, spec, status and task log.

## Review

Local review only; no independent agent tool available.

## Changelog

Record removal of the separate correction frame pipeline.
