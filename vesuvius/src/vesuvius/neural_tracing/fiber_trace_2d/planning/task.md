# Anisotropic and direction-guided training refinement

Use normal bending weight0.1, tangential weight10.0 and direction weight1.0
(user follow-up: increase tangent and direction by10x, leave normal unchanged).
Keep split loss logs/shared raw-EMA targets, fixed crops, free test ends and caps.
Stop previous run and launch fresh geometry from original non-refinement11800.
