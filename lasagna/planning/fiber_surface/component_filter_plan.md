# Connected-component support filter

Keep whole connected components with at least three distinct source fibers and
both H and V, using the saved deformed crossing graph. This is not a clique or
a minimum-degree filter. Count source identities, not split pieces. Filter before
sampling, ruled initialization and fitting; do not mutate upstream data.

Reuse source_components. Reject partially selected crossing edges; omitted whole
components must not trigger missing-crossing errors during metadata preparation.
Report source/piece/component/crossing counts and retained source identities.
Default connected-HV minimum is 3; explicit minimum 0 disables the filter for legacy replay.

## Validation
Test singleton, pair, H/V triple, transitive chain, repeated pieces, invalid
family and partial edge selection. Run regression tests and real deformed fit
using the existing approved managed runner, new output directory.

## Spec update
Connected-HV selection now additionally requires mixed-family components of the
configured minimum distinct-source size. Legacy input remains immutable.

## Docs updates
Document selection setting and measured counts in docs, spec, changelog and
active task log. Archive the previous run log; update task/status.

## Requested filtered visualization
Reuse the reviewed connected_solve_plan renderer without code changes. Match the
previous global UV band: old origin U19700, columns244:833; filtered origin34100,
so new columns100:689. Render CT JPG,8px H/V overlay/composite LZW TIFF and cropped
inspection tifxyz. Validate coverage, image alignment and input paths; no refit.
