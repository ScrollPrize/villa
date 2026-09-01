# Task: separate winding value and sign-hardness constraints

Separate signed winding observations into independently weighted winding-value
and sign-hardness constraints throughout extraction-to-BP materialization,
scoring, diagnostics, and reference benchmarking for both dominant
perpendicular and parallel relations.

The ordinary winding-value residual remains signed. Sign hardness adds the
separately tunable high or hard reversal penalty. All
constraint listings, agreement checks, and reference benchmark counts must
report winding value and sign hardness separately, without allowing a zero solver weight to
remove the corresponding item from the benchmark denominator.

After validation, tune the seven weights on the established 1024 reference
benchmark and promote the selected defaults.
