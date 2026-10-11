# CCW versus reference visualization (archived)

- Plan uv_fullpass151_vis_plan.md independently approved.
- Reuse established TIFF/SVG renderer and validated geometric projection cache;
  CCW saved10-pass state only.1000px high,2px opaque curves, equal physical scales.
- Explicit global_winding_offset=11 aligns colors with the equal-fiber L1 metric.
  Historical default length-weighted offset remains unchanged when not supplied.
- Approved runner configs run_reference_fullpass151_ccw.json and
  report_uv_fullpass151_initial_reference.json launched CPU-only.
- Initial/final q10/q90 recoverable; intermediate states absent, no fake ranges
  and no optimization rerun.15related regression tests passed.
- User additionally requests explanation of broken logits after these results.
- Rendering completed (exit0); all three TIFF/SVG maps plus reference TIFF and
  detail TIFFs. Initial quantiles49/122(+12), final48/121(+11), reference42/124.
  Results/interpretation: uv_fullpass151_vis_results.md.16tests passed, including
  explicit offset override and historical default preservation.
