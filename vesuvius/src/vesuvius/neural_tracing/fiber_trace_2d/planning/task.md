# Task: accelerate anchor local refinement

Optimize the measured anchor-fitting bottleneck in
`refineLocalComponents()`. The version-2 profile attributes 86.3% of anchor
fitting worker time to local component refinement and records 32.49 billion
logical observation visits there on the canonical 5,000-base-voxel replay.

Evaluate the initially proposed component-scan fusion and normalized-direction
reuse, but retain only changes that improve measured performance while
preserving fitting decisions, double precision, compensated accumulation
order, deterministic artifacts, and profile semantics. Use the profile to
identify a behavior-preserving alternative if those proposals regress.
