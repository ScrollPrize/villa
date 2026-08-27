# Task: Use parallel and perpendicular constraints in fiber BP

Allow binary and Mixed-state fiber belief propagation to consume the full
selected no-split constraint graph rather than requiring perpendicular-only
links. Preserve score decisiveness through the existing same/different energy
gap, and make all consistency diagnostics aware of whether each merged factor
prefers equal or different H/V labels.
