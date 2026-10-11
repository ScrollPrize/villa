# Reference winding comparison — 100-sweep solutions

Completed2026-10-11 with run_reference_comparison100.json through the approved
benchmark runner. Final successful run705.94s (11m46s), CPU-only. No solve or
inference rerun. Shared projection:8,129,117curve samples,11,713,673valid stride2
reference cells. Source parameters and full input hashes are in output summary.

Output root: `/home/hendrik/villa2/global-fiber-sgd/data/reference_comparison100_v1/`

| Metric | CCW100 | CW100 |
|---|---:|---:|
| Retained pieces |395,549|395,118|
| Retained original fibers |395,547|395,117|
| Original fibers with chosen-winding projection support |304,778|304,372|
| Retained length with nearby reference support |79.7334%|79.6955%|
| Retained length supported on median chosen winding |78.4301%|78.3939%|
| Matched sample median distance, native voxels |14.9318|14.9254|
| Matched sample p95 distance, native voxels |47.6402|47.6027|
| Single global integer numbering offset |+38|+38|
| Arc-weighted median absolute nearest-reference winding difference |12.9936|12.9431|
| Matched length within1continuous winding |4.0205%|4.0823%|

Coverage denominators are retained source arc length, not sample count. Distance
quantiles are sample quantiles. Difference statistics use local nearest-reference
continuous q after the single global offset, not the restricted projection or a
per-fiber optimized alignment. These are rough reconstruction disagreements,
NOT anatomical accuracy scores. Rectangles provide approximate winding labels;
projection uses bounded nearby triangulated cells, not exact global bilinear
projection. Nevertheless the comparison shows substantial winding disagreement
that a single numbering offset does not remove. No local warping/alignment was
applied to hide it. The reference does not cover the full native scan height;
uncovered or distant geometry is not forcibly assigned.

## Images

Common `reference_own.tif` colors the valid reference grid by its own continuous
rectangle-derived q. For each `ccw/` and `cw/`:

- `fit_own.tif`: our layout, our winding colors.
- `fit_reference.tif`: our layout, reference winding colors; unsupported gray.
- `reference_fit.tif`: actual projected fiber curves on their median reference
  winding, colored by our fitted winding. Unsupported sections omitted. This is
  the requested discrete-choice/reference-layout view, not surface repainting.

Corresponding SVGs are fully vector but large (~482–616MiB); TIFFs are the
practical first inspection files (~28–30MiB per fiber map). Full images2000px
high with equal physical axis scale, no titles/histograms/margins. Fit widths
108962(CCW)/105507(CW); reference-fiber maps119129px. `_detail.tif` files are
central crops for inspection, not substitutes for full maps. `palette.csv` gives
integer colors. `fibers.csv` retains original source IDs and solved-piece arcs.

## Verification

15tests passed including synthetic plane/edge/degenerate projection, source-level
weighted median shared across split pieces, endpoint label limits, palette
interpolation, actualU inversion, sampled gap guard, end-to-end TIFF/SVG/JSON
export and validated projection-cache reuse. Both real renders completed exit0;
reference and both solution detail images inspected. git diff --check clean.

Command:

```bash
/home/hendrik/.venv_las/bin/python -m pytest -q --noconftest -o pythonpath=/home/hendrik/villa2/vesuvius/src vesuvius/tests/test_fiber_reference_comparison.py vesuvius/tests/test_fiber_reference_surface.py vesuvius/tests/test_fiber_uv_visualization.py
```

Independent plan and code reviews complete. Known limitations: center-labeled
cells can lose support near winding boundaries; the sampled connector hole
guard can miss tiny diagonal invalid-cell crossings. Approximate matching may
choose a nearby wrong physical sheet. No exact reference-winding sidecar exists
in the downloaded prefix. These limitations are explicit, not optimizer changes.
