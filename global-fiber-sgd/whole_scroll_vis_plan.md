# Whole-scroll retained-fiber preview

User requests about100fibers across whole scroll, no model-marked broken fibers.
Add deterministic spatial farthest-point selection of retained fibers using AFV
bbox centers normalized by each axis extent so height does not dominate. Select
100IDs, preserve full original curves, reuse fixed OBJ/PLY exporter and arrows.
No optimizer, inference, normal changes or old preview overwrite. Record selection
method/IDs and exclude broken explicitly. Spatial overview, not random/unbiased
statistical sample. Test deterministic spread and count; validate export counts,
absence of broken fibers and original polyline segment totals. Approved runner.
Spec/docs: visualization-only option and new output path in sign_resolution.md;
log/status/changelog; independent review per AGENTS workflow.
