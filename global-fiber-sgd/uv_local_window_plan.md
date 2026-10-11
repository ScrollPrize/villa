# Profile quadratic and trial local winding windows

Updated user requirement: all measurements and quality runs use exclusive GPU
allocation. Quality GPU1-4, profiling GPU5 sequentially; shared pilot timings
are not final comparison evidence. Preserve interrupted pilot outputs separately.
Manager exclusive requests are globally serialized: use one exclusive suite
lease and per-GPU subprocess lanes under it. Existing solver module is reused,
not copied. Child jobs do not request another lease. Validate unique outputs,
exclusive allocation and lane count; collect errors and logs. No solver changes.

Profile current corrected compiled squared loss using existing pair benchmark
and short production update timings. Report pair FW/BW versus whole update,
not mislabel their difference as other losses alone (optimizer and coordinates).

Trial first sparse variant: keep full parameters; evaluate distance/V terms only
in +/-N bucket indices around each endpoint's MAP, N=2,4,8,16. Recompute each
update. Renormalize local probabilities to full retained mass so discarded mass
cannot evade cost. This changes the soft objective, explicitly experimental.
Broken remains unchanged. No constraint-proposed windows or placeholders yet.
Sign hinge remains exact over all hypotheses via sorted prefix mass/moments;
sort coordinates because learned offsets can reorder buckets. Unknown sign zero.
Preserve all unary/continuation losses. Reject unsupported fixed-sign-charge mode.

## Validation
Also benchmark B=4096,8192,16384,32768 for full and N2, with CUDA phase timers
for coordinates, pair forward, other loss forward, backward and optimizer.
Measure peak allocated memory and constraints/sec, then run best-throughput
larger batch to comparable event count (fewer optimizer updates explicitly).
CPU float64 loss/gradient oracle for prefix sign, signs +/-/0, unordered/tied
coordinates and zero masses. Local full-range matches full squared; boundary
windows exclude clamped duplicate indices. CUDA compiled/eager validation and
timings. Run matched fresh CCW151/4096/16130 sampled updates for each N via
approved GPU-managed runner. Compare prior corrected full baseline, reference
L1/within2, retention, sign audit, ranges, time and captured window mass.

## Spec update
Opt-in local_window_radius; defaults remain full. No new physical evidence.

## Docs updates
Archive prior task log; current task/status/log, commands, profile/results and
changelog. Independent review before implementation and after tests.
