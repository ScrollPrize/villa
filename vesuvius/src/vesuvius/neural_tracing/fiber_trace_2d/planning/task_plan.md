# Plan

1. Independently review seam admission and downstream alternative semantics against specs.
2. Remove percentage admission, retain physical spatial/direction compatibility, and instrument endpoint eligibility and candidate rejection reasons with explicit denominators.
3. Fix downstream handling of unmatched and alternative continuations without treating all alternatives as simultaneous hard equalities. Preserve topology and report any unsupported semantics explicitly.
4. Add regression tests for unequal overlap lengths, exact 32-voxel threshold, ambiguity, unmatched endpoints and accounting.
5. Build/test via approved runner; reassemble saved local-dedup tiles only in a fresh output directory. Report old/new matches and rejection breakdown.

## spec update
Absolute shared support threshold; explicit auditable seam consumption policy. Preserve tracer and prediction settings.

## docs updates
Update tiled-tracing benchmark with complete prior run, current matching rules and audited results.

## changelog
Archive completed tiled tracing; record seam repair and tests.
