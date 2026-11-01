# No initialization-width prior

Initialization widths are guesses, not observations. **Do not re-add a loss
pulling fitted winding widths toward initialization**, including a radius-based
initialization. Do not expose such a term to tuning or restore it from a recipe.

The former `width_prior * mean((width / initial_width - 1)^2)` is deleted from
the soft/probability objective, legacy layout solver and diagnostic objective.
Nonzero legacy settings are rejected. Zero-valued historical settings remain
readable but have no loss implementation.

Width smoothness and the soft minimum-width penalty are separate, valid terms.
Initialization may supply starting values and numerical scales, never a target
fitted width. Historic results with the deleted prior are not valid baselines.
