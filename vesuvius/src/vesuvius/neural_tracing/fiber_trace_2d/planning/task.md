# Task: Post-solve perpendicular consensus

After the no-split, perpendicular-only direction-ablation MILP, initialize one
continuous H value per fiber from its discrete result (H=1, V=0, Broken=0.5),
then synchronously update values from confidence-weighted perpendicular
neighbors for a configurable number of iterations. Write exactly those
represented no-split fibers as ten short-named 0.1-value-band OBJ layers
alongside the main output.
