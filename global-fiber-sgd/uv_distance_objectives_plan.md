# Four matched global-distance experiments

User requests current likelihood baseline plus expected squared, expected Huber,
and expected Huber + likelihood. All use fused kernels, fresh non-reference radial
initialization, random minibatch updates, GPUs0–3 respectively, reference only for
diagnostics. Reuse prior sign/HV/direction evidence; no earlier stages rerun.

## Specification update
Add opt-in distance_objective (likelihood default, squared, huber,
huber_likelihood). Preserve all other terms and all151 hypotheses. For expected
variants let r=(ordered_distance-measured_distance)/distance_sigma, including
signed distance without clamping wrong signs (unknown sign uses absolute delta).
Squared is .5*r²; Huber is .5*r² for |r|<=1 and |r|-.5 outside. Multiply by full
unconditional assignment pair mass, allowing broken to pay its unchanged cost.
Squared/Huber replace likelihood with expectation of distance+existing metric
cost. Hybrid adds expected Huber to the unchanged likelihood, metric once.
Keep exact existing sign loss, continuation, priors and fixed rejection10.
Initial distance weight1, Huber transition .5turn (sigma.5), hybrid weight1.
These are initial matched tests, not a tuned sweep; objectives across modes have
different meanings/scales and must not be ranked directly against one another.

## Run and validation
Use CCW, K151, batch4096, seed0,16130random updates (~10event-pass equivalents),
log every1613. No global accumulated gradients, no warmup, no freezing, no warm
start/reference initialization. Shared24GiB GPUmanager grants pinned0–3.
Test increasing wrong distance, gradient toward compatible probabilities,
broken gating, boundary Huber and unchanged likelihood. Check fused/eager loss
and gradients for new modes through managed benchmark before long runs.
Report reference offset-corrected L1, within2%, retained arc%, sign%, q10/q90 vs
reference each logging interval. Save configs, logs, full final audits, outcomes.

## Documentation
Archive prior task log; current task log, status, changelog and results report.
Independent plan/code review under the existing AGENTS workflow. No new runner
or dependencies. Do not implement band pruning in this task.
