# Reference-initialized full joint run and affine-V comparison

- uv_reference_init_plan.md reviewed. CCW151,10fullpasses,reference-seeded logits,
  unmatched radialfallback,all globalvariables trainable from firstupdate.
  Rejectionwarmup0,fixedcost10,entropy_ramp=false,no strong follow-up.
- Approved runner run_uv_reference_init151_ccw.json launched;23tests passed.
- User clarifies vis: preserve completeUVgeometry, only approximate global
  vertical alignment. PreviousphysicalZmode changedshape (disclosed).
  New plan reference_affine_v_plan.md; no pointwiseZreplacement or cropping.
- Affine plan approved independently;15 comparison/rendering tests passed,
  including reversed affine slope and nonmonotonic physical Z preserving the
  original UV rows. Approved runner run_reference_fullpass151_ccw_affine_v.json
  launched, CPU only. All reference rows and fitted extents included in canvas.
- Reference-initialized pass2: L1=0.785325, within2=95.3285%, retained length
  86.0172%, fixed100k sign violation diagnostic5.5369%; offset0, q10/q90=42/124
  for both fit and reference. No parameter freezing or delayed broken updates.
- Affine render completed successfully in283.31s. Output
  data/reference_fullpass151_ccw_affine_v_v1; CCW maps all complete. Shared
  canvas0..80000/1000px; reference V'=0.8189717545*V+12456.62865;
  fit V'=0.9982529628*V+59.37753. Full UV geometry preserved up to affine scaling.
- User confirmed visualization now present. Proposed extra root symlink aliases
  were not implemented; existing completed fit_reference/reference_fit are
  together in ccw/. No data was moved or duplicated.
- Pass5: objective33.92899, sign2.41769%, retained83.96633%, L1 .80763,
  within2 94.96592%. Fit/reference q10/q90 remain42/124,offset0.
- Completed10passes successfully,exit0; final results and every pass preserved
  in uv_reference_init_results.md. Final objective3.255119,L1.880643,
  within2 93.625309%,retained83.869947%,full-event sign violations1.499490%.
  No further visualization was run after the user's instruction to report only.
