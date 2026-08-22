# Task: diagnose persistent wide-radius fiberlet replay failures

Use a small extraction radius to obtain a matched, correct fiberlet replay, then
compare it with the best radius-768 replay's two persistent failures.

- Preserve the completed larger-lookahead sweep results in durable documentation.
- Run the same replay objective and search settings at a small radius and confirm
  whether it follows the reference through the two wide-radius failure regions.
- Record decision diagnostics for both runs and identify where their selected
  routes first diverge, which alternatives were retained, and which edge, join,
  or weighted profile costs cause the wide-radius search to prefer the wrong path.
- Assess whether route collections can support offline parameter tuning without
  rerunning complete traces. Distinguish objective-only rescoring from search
  parameters that change which route candidates are generated.

## Deferred evaluation options

- Replace the smaller-radius proxy with an oracle over the same wide-radius
  graph that minimizes the ordinary replay objective while constraining route
  geometry to the reference threshold. The constraint must use the configured
  Lasagna-normal ellipsoid, not an unrelated isotropic corridor.
- Compare the unconstrained winner with that best admissible route from the
  same seed, checkpoint history, graph, and cost model. Report both geometric
  error components and the complete ranked objective decomposition.
- Later evaluate a tolerant correctness rule that permits a route to leave the
  reference threshold temporarily when it returns within a bounded base-arc
  distance. This needs explicit limits on excursion length and severity so it
  does not hide a real fiber switch when the annotated reference is accurate.
