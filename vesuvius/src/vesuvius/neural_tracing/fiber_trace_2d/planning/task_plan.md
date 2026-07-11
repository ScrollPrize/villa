# Task Plan

## Spec Update

- Document that contrastive negative candidates are restricted to valid pixels
  inside the CP-neighborhood reachable rectangle implied by
  `augment_shift_x/y`; unreachable edge pixels are ignored, not negative.

## Code

- Add a small helper that builds a patch-shape mask for reachable contrastive
  negative pixels from patch size and configured shift bounds.
- Pass that mask into the contrastive embedding loss from training and benchmark
  contrastive paths.
- Keep the existing deterministic negative pairing scheme, but apply it only
  after `valid`, non-CP, and reachable-region masks are combined.

## Docs Updates

- Update `docs/code_structure.md` contrastive training notes with the edge
  exclusion behavior.

## Changelog

- Add a short changelog entry for the contrastive negative edge-mask fix.

## Tests

- Add a focused unit test where high-similarity non-CP pixels exist only in the
  unreachable edge region and verify they do not contribute to negative loss.
- Run the focused fiber trace 2D loader tests.
