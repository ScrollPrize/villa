# Reference winding comparison

## TIFF-only correction

User requires TIFF raster outputs, including detail crops. Change preview writer
to LZW TIFF. Add maintenance operation to existing runner converting only this
experiment's *_detail.png files into pixel-identical verified LZW TIFF, then
unlink those PNG copies. Do not rerun projections or change full images/SVGs.
Test compression, exact pixels, and cleanup. Docs/spec update: all raster outputs
TIFF; correct results documentation and record correction in task log/changelog.

Use rectangle overlap column midpoints as half-integer winding landmarks,
piecewise-linear continuous winding evidence, deterministic integer colors.
No correspondence optimization. Project actual AFV curve samples onto nearby
reference geometry (XYZ scaled by four), take an arc-weighted median discrete
reference winding per original AFV fiber (shared across solved pieces), then project that fiber to its chosen
winding only. Preserve unsupported points as gaps. Save assignments, distances,
coverage and provenance. Compare both saved 100-sweep solutions without edits.

Produce reference-own, fit-own, fit-reference-colored, and reference-layout
fiber curves colored by fit winding. Last view is discrete per-fiber assignment,
not nearest-fiber painting of a surface. Shared palette and one recorded global
integer gauge offset per solution; no local alignment or optimization. Respect
equal physical UV scale and use compressed TIFF plus SVG fiber views.

Implementation: reusable geometry projection with scipy spatial index and local
surface approximation; bounded batches, shared geometry across CW/CCW. Reuse
canonical solved coordinates, AFV reader and existing tiled TIFF renderer.
Tests: rectangle landmarks, palette continuity, weighted median, known planar
projection, discrete winding selection, missing/invalid geometry. Validate real
data counts and projection coverage; report approximations explicitly.

Spec update: diagnostic read-only reference comparison; winding rectangles are
heuristic, not exact reference labels. No modification to optimizer or extraction.
Docs updates: replace obsolete continuity-optimization proposal, record command,
outputs and limitations in reference_comparison.md, task log and changelog.
