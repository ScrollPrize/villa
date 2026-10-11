# Completed 100-sweep fiber maps

Render both saved solutions without optimization or GPU allocation. Reuse existing
piece selection/geometry, confidence opacity, colors and winding boundaries.
Raster maps use the entire 2000-pixel height for the map, equal U/V scale, no
surrounding labels or histograms. Export histograms separately. SVG and true vector
PDF retain annotations and histograms. Use installed matplotlib for vector PDF,
sharing drawing operations; scale PDF page uniformly to a portable page size.

Spec update: presentation only; no solution/selection changes. Add CPU layout-only
entry point and explicit render options. Tests: TIFF dimensions/compression,
map-only bounds, histogram files, vector PDF structure, existing regressions.
Docs updates: result links, counts, sizes, run command and verification in task log
and results doc; archive prior log. Independently review plan and implementation.
