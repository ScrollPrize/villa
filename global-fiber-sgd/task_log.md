# Corrected quadratic fused run

- Prior corrections committed: c468354dd;81 tests passed.
- Independent plan/config review approved.
- Fresh non-reference CCW151,4096 random minibatches,16130 updates.
- Only loss corrections differ from earlier squared run; V prior .01/1000.
- Config: run_uv_squared_vertical_split.json. No extra stages or visualization.
- Completed exit0; all10event-pass equivalents. Solver699.28s versus712.58s,
  nonisolated shared GPU timing. Final L1=11.723346,within2=13.213415%,
  retained length66.016892%,sign violations3.330811%. Effectively unchanged
  accuracy versus historical squared. Full progression and exact command in
  uv_squared_vertical_split_results.md. No moments acceleration implemented.
- Managed task595e3514-3793-404a-aee0-75bb5b698c45 granted shared GPU1/24GiB
  after training released capacity. Approved runner active, initial reference
  L1=12.2423449311,within2=12.991196442%,offset=12,fit q10/90=49/122.
