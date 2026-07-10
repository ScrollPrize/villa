# Task: Fused Forward/Backward Augmentation Coordinate Maps

Refactor geometric augmentation coordinate handling.

- Make the coordinate transform handling generic: each geometric transform
  should expose forward and backward map/application functions.
- Compose transforms into one fused forward mapping and one fused backward
  mapping instead of maintaining separate formulas for sampling, line/CP
  generation, and tracing helpers.
- Remove the smooth-offset nearest-output-pixel search. Smooth offset is a
  vertical column-dependent displacement and should generate/apply both maps
  directly.
- Adapt line generation to start from cached source `line_xy`/CP coordinates and
  apply the fused source-to-output augmentation transform.
- Preserve the existing image sampling semantics: final image sampling still
  uses one output-to-source map and samples the volume once at final coordinates.
