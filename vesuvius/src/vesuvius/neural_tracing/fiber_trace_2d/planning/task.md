# Task: restore baseline fiberlet search around weighted lookahead

Fix the regressions introduced by checkpoint-relative subsegment scoring.

The fiberlet graph replay must remain identical to the prior search design except
for how cost is integrated from the current checkpoint through the lookahead:

- Preserve the accumulated route cost before the current checkpoint.
- From the checkpoint onward, integrate stored subsegment costs with the
  configured integration step, delay, and geometric falloff.
- With weight one, behavior should remain close to the old baseline. Decoded
  subsegment costs remain the source of truth and should naturally add to the
  whole-fiberlet cost apart from codec and floating-point rounding; do not
  rescale them to force exact equality.
- Cost integration must be incremental and cheap. It must not repeatedly walk
  complete route histories or rescan complete segment profiles.
- Exact and bounded search must retain effective admissible pruning under the
  new objective.
- Use one general integration algorithm for every weight. At weight one,
  changing integration spacing may cause small interpolation or accumulation
  differences, but must not cause a material route or failure-count regression.
- Validate quality and performance against the actual pre-change baseline, not
  against a new-profile control mislabeled as baseline.
