# Combined coordinate gradient experiment

- User approved implementation, measurement and documentation. Plan:
  uv_coordinate_plan.md; independently reviewed by reference_init_review.
- Opt-in coordinate_backend=combined. Explicit width gradient includes physical
  U prefix for continuation losses; endpoint-q prefix cancels analytically.
  CUDA FP32 tiled fixed-order reductions, no atomics; CPU/double small reference.
  Generic baseline remains default. First derivatives only; derived-boundary
  contract documented. Same-bucket algebra changes FP32 rounding, not objective.
- CPU regression: 9 passed; managed CUDA preflight checks values/gradients and
  bitwise repeatability before every combined run (seed reset afterward).
- Exclusive single-GPU sequential baseline/combined B4096/B16384 benchmarks and
  profiler launched using run_uv_coord_matrix.json through approved runner.
- Reusing anchor calculations across three unary points is not implemented yet;
  first measure combined derivative reduction in isolation.
- Independent implementation review approved derivative/reduction/integration.
  Regression101passed (CUDA preflight separately managed); no new dependencies.
- All6short runs completed exit0, exclusive GPU5.4096 mean23.415->14.390ms,
  16384 mean60.273->21.492ms (1.627x/2.804x). Generic indexed backward27calls/
  10.312ms ->3calls/.182ms; replacement7coordinatebackwards total.983ms.
- Matched4032-update16kLR.06 quality solve started via approved runner;
  exclusive3067633e-45dd-4daf-8b7b-036e1d1a4fbf. Defaults still unchanged.
- Reviewer inspected generated PTX: ordinary Triton division emits div.full,
  not div.rn. Corrected both divisions to explicit tl.div_rn. v1combined runs
  are preliminary; queue v2 timings/profile/quality. Existing running v1process
  already loaded its kernel and finishes unchanged; preserve output provenance.
- User explicitly authorizes ordinary/faster division after quality validation.
  Added coordinate_division=standard|precise so both are reproducible rather
  than replacing code between experiments. RN configs explicitly precise;
  original combined configs explicitly standard. Final default choice pending
  completed comparison. Sampling/loss/optimizer settings unchanged.
- Standard fullquality completed:213.333s recorded solver time,referenceL1
  11.659781,within2=13.287325%,retained65.490114%,sign3.195105%. Control
  11.660019/13.292521%/65.485704%/3.165387%; not bitwise, small aggregate changes.
- Precise short timings:4k14.121ms,16k21.994ms; ordinary14.390/21.492ms.
  Do not claim ordinary always faster. Reviewer confirmed generated precise
  PTX has2div.rn.f32; historical standard has2div.full.f32.
- User adds next steps: larger-batch benchmark after this validation; then
  implement a single distributed solve across availableGPUs0–5. Not independent
  replicas. Must retain exclusive benchmarking and report globalbatch/LR.
- Precisequality completed217.483s;L1=11.660387,within2=13.263416%,retained
  65.485030%,sign3.179760%. Standard/precise trajectories comparably close to
  reference control. Standard division selected for combined path per user;
  reference backend still default. Both exclusive tasks released. See results.
