# Position weight0.5 follow-up

User requests metric position loss half of bucket loss instead of0.1.
Config-only matched fresh/ref10k trials: stateLR0.07348469228349533,widthLR2,
151hypotheses,same subset/initializations/seed/globalbatch24576,sixGPUs0–5.
No production code, sign, regularizer or continuation changes. Prior result
and full logs committed27a8a5eb8; tables in uv_bucket_split_results.md.
Plan: uv_bucket_split_plan.md follow-up. New configs run_uv_bucket_half_*.json.
Independent config review approved;9 split-objective tests passed2.52s.
Approved suite launched, session3822, manager task
81363a56-ed6c-4634-86a7-3f50e6637e2e. Cooperative release completed and sixGPU
exclusive grant acquired. Fresh production compiled/eager sixGPU gradient and
coordinate checks passed. At3k loss4.8949,L1=2.0661,retained38.8721%, versus
0.1weightL1=2.2833 withsimilarretention; preliminary improvement only.

Both10kcomplete,suiteexit0,finitefinalstates,managerfinishedheld0. Fresh172.01s,
reference171.21s; sixGPUcompiledchecks passedboth. Freshloss3.70636,L1=1.96746,
within2=68.74576%,retained50.17743%,width.913733. Referenceloss2.58818,
L1=.642303,within2=97.89590%,retained72.82328%,width.895079. Bothzerobucket
violations. Refdriftlatewithoffset0: not anoffsetswitch. No objectivepromotion.
Full1ktables,oldbaselines,commands,timing inuv_bucket_half_results.md.
