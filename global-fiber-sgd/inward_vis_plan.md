# Inward normals / outward arrows

Keep the optimizer's outward internal gauge and historical statistics unchanged;
canonicalize saved solutions to inward by globally negating signs/logits and
complementing probabilities. Mark convention explicitly; legacy NPZ is outward.
Use shared conversion for future export/solver saves. Do not flip independently
per point (would destroy continuity in folds). Weak radial prior is not a guarantee
of anatomical orientation at every point.
Regenerate existing long-preview PLY from its indexed points, preserving selection
and OBJ bytes. Store inward normals and separate outward display directions.
Arrow length640/radius20, five times previous. Write inward solution copy, preserve
historical outward solution, update index hash/path and README. Atomic file replace.
Tests: conversion idempotence, signs/probability/logits, arrow tip/radius and
unchanged OBJ; approved existing crop-benchmark runner. Log/docs/status updates.
