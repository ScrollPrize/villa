# Stronger fiber UV-offset regularization

User wants smaller, substantially smoother offset changes between locked H/V
crossings, without removing their freedom to slide. Reuse existing offset losses:
displacement weight0.01→0.1, offset first-derivative weight1→10 and100. Keep stretch,
limits, UV learning rate, mesh weights, initialization, input cohort and800steps
unchanged. These penalties act on added offsets, not curvature of original UV.

Run both matched fresh solves through approved runner, nonexclusive8GiB grants
on GPUs0–5. Compare baseline and candidates: absolute UV shifts, RMS/p95 offset
slope and change in slope along arc, H/V normal fit, positional error,folds,timing.
Check crossing equality and unchanged initial samples. Render strongest useful
result with existing CT/HV renderer, same global UV bounds and8px lines.

## Spec update
No formula/default API change; candidate configs explicitly specify stronger
sliding_settings. Keep historical baseline reproducible. Regularization remains
soft and does not guarantee an unfolded surface or straight fibers.

## Validation and docs
Run sliding/connected/render tests, independent plan review, matched data audit.
Archive prior task log; record settings, results, commands and selected candidate
in active log, docs, changelog and status. No new solver or loss implementation.
