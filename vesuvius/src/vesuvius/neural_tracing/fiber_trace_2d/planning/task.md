# Task: Align Lasagna normals with belief propagation

Use the current fiber BP message-passing scheme to resolve the sign ambiguity
of normals sampled from a regular Lasagna `grad_mag`/`nx`/`ny` manifest.

- Keep the alignment as reusable core functionality for later H/V BP work.
- Also expose it as a standalone command over an explicit base-voxel bbox.
- Visualize the same sampled normals before and after alignment as separate OBJ
  files. Each normal glyph has a short crossed base and one directed stroke.
- Do not use or modify the legacy NormalGridVolume alignment path.
- Parallelize the standalone alignment BP so large normal lattices can approach
  sub-second solve time on a many-core GCC/OpenMP build, without changing BP
  factors, iteration semantics, convergence, or results.
