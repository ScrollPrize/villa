# Fixed sign charge task log

- Prior shared-anchor task preserved in uv_anchor_results.md, full/pilot reports,
  uv_layout.md and changelog; no previous output overwritten.
- User: retain magnitude penalty but add fixed charge sufficient to correct or
  disable (>50%broken) any wrong selected sign. Plan:uv_sign_charge_plan.md.
- Explicit selected-pair safeguard avoids dilution by100hypotheses; existing
  joint-argmax broken readout replaced only in new configs by requested >50%rule.
- Independent plan/code review approved normalization and sampling. Fixed two
  warm-start findings: inverse-softplus width restore and respecting pinned SHA.
- Charge C=gamma*(E/N)*R*min(ca,cb), c=.5+.5*degree/mean_degree.
  Selected wrong pair is charged at full retained-pair mass, not bucket mass.
  Fresh float64 selected geometry; no stale forced violations/posthoc deletion.
- Pilot4:1500updates,1024events/update,50%active mixture with exact importance
  weighting.6525→227 strict violations;95.6542→69.4125%arc;18.8906%pieces
  rejected.29.67s solve. Fails zero-violation requirement. Pilot16 queued through
  manager, temporarily waiting for unrelated exclusive test on GPU0.
- Pilot4 predates warm-restore fix, but all saved widths exceed2500, where the
  inverse-softplus equals subtraction at this precision; no numerical impact.
- Pilot16:6525→52 strict violations,61.2577%arc,25.8971%pieces rejected,
  29.74s. Full CCW/CW16 trials launched (4000steps,batch4096); additionally
  lower-learning-rate pilot continuation checks whether residual errors settle.
- User clarified227remaining constraint violations can be acceptable for this
  experimental stage; eventual filtering possible. No posthoc filtering applied.
  Added endpoint broken-probability diagnostics (unique pieces, pair maxima,
  histogram, retained population comparison), requested before further decisions.
- Requested pilot4 diagnostic:227violations touch215unique pieces/fibers.
  Endpoint broken probabilities mean3.2214%,median0.2733%,p95=19.3816%,
  max48.8573%. Higher endpoint per violation:mean6.2727%,median1.2620%,
  p95=26.2682%.109/227pairs have both endpoints below1%;182/227below10%.
  Thus most remaining bad pairs are confidently retained, not near50%.
  Of6525initial violations,5271disabled,1203corrected while retained,51remain;
  176newly violated edges make final227. Indicates changing assignments as well
  as unresolved initial errors. Diagnostics saved in pilot4/audit.json and
  uv_sign_charge_pilot_results.md/.json. Probabilities are fitted, not calibrated.
