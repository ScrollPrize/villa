# Per-check rejection audit

Instrument shared local-event extractor without changing candidates, decisions,
deduplication or outputs. Count mutually exclusive first-failure geometry reasons:
missing/degenerate tangent, seed/refined/probe orientation, endpoint support,
central/probe distance, central/probe reciprocal, monotonic correspondence.
Count winding-invalid/distance-tolerance/category/sign failures separately; also
record overlapping winding failures for candidates reaching winding certification.
Geometry checks requiring a valid encounter cannot sensibly be evaluated for all
earlier failures; report these as first-failure attribution, not independent
leave-one-filter-out recovery. Dedup counts inferred only after exact accounting.

Test existing native suite plus slab run: preserve all existing aggregate counts.
Then run full unsigned extraction with no database export into a new directory.
Verify identical full counts; table denominator initial+refinement candidates.
Do not overwrite reusable constraint DB. Use approved build/evaluation commands.
Spec: optional diagnostic counters only, no extraction semantics change.
Docs/log: persist audit table, denominators, overlapping-test limitations and commands.
