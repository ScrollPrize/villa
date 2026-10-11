# One fresh joint solve, strong ordering throughout

User correction: only run the global winding/U/V solve. No staged weak solve,
sign refinement, or categorical repair. Upstream normal/H/V/direction results
stay fixed. The previous staged experiment is superseded; no process remains.

## Run

Use existing tested joint Adam solver, 151 winding candidates plus broken,
fresh radial initialization (no U/V checkpoint). All geometry, widths, winding
probabilities and rejection variables train together from step zero.
Use sign_charge_multiplier16 from the existing strong ordering formulation,
rejection_warmup0. No later penalty change. Both CCW/CW, GPUs0/1 independently.

Ten event-pass equivalents means ceil(10*6,606,100/4096)=16,129 minibatch updates.
Sampling is with replacement and importance weighting (active_fraction0.5),
so this is an equivalent sample budget, not ten exhaustive sweeps. Report that
distinction explicitly. No early feasible stop; record diagnostics about once
per equivalent pass. Keep existing batch4096 and learning rates. Reserve24GiB
per GPU (previous joint/sign run observed ~12GiB); approved runner only.

## Spec update / tests / docs

No optimizer implementation change. Configuration forbids warm starts and staged
handoffs for this experiment. Existing40tests already cover fresh151/trainable
geometry and loss paths; validate exact configs and independent plan review.
Audit final full-graph results and write task log/results/status/changelog.
Store under data/uv_joint151_strong_v1, preserve but do not promote earlier runs.
