# Corrected mini5 reference

Replace mini5 piece000 with the user's corrected annotation, preserving historical
manifests and results. Freeze corrected input in a new benchmark revision with
source hash, prior piece identity, coordinate metadata and forward direction.
Use native replay-references to calculate eligible CP-bounded length (exclude
extrapolated tails and Lasagna spans); do not use full line length. Keep other
four inputs byte-identical, same model/cache/weighted4 settings. Rerun with vis
in a fresh output directory and compare five cases, log results.

## Spec update
Explicit benchmark corrections create a versioned manifest; never silently
mutate locked historical datasets. Validate corrected source hash and eligibility.

## Docs and tests
Unit-test metadata restoration/hash checks; use existing approved benchmark and
test commands. Persist corrected manifest, native reference plan, results and
before/after table. Cost oracle remains paused.
