# Bucket topology / metric position split

User requests bucket-only sign and distance, old positional distance at0.1,
joint scratch/ref-init runs at newly adopted high LR. Interpreted10 as10k
passes matching prior comparison; told user. Prior results preserved in
uv_class_balance_lr_results.md; commit18b5cce30 adopts default LR.

Plan uv_bucket_split_plan.md. Independent review flagged anchor-relative
seams. Added fixed integer correction from recorded unwrapped endpoint and
anchor angles, explicitly disclosed. No learned phase, offset or width enters
topology. Existing priors, continuation, vertical loss and broken updates stay.
Implemented opt-in squared/full-hypothesis split and matching bucket audits.
Fixed charge uses selected buckets.55 focused tests passed9.20s. Review approved
core implementation; added mixed-shift distributed preflight and robust anchor
integer recovery rint(raw_anchor-fractional_anchor), including seam test.

Final added seam test plus layout/anchor/localwindow/distributed regressions:
41passed9.73s (total89 across focused/new and broader suites). Approved runner
launched run_uv_bucket_split_suite.json, session3087, manager task
7b42a4cb-46be-4e0e-af0e-a0d4a98d13e8. Cooperative training release completed;
sixGPU compiled/eager gradient validation passed, coordinate value/gradient
and bitwise repeat checks passed. Fresh started,1k retained32.94%,2k36.57%,
zero bucket signs but poor retention explicitly reported, no early victory claim.

Both runs finished, suiteexit0. Fresh166.28s/ref169.70s; finite saved state and
width arrays; managerfinishedheld0. Freshloss3.6809,L1=2.1762,retained50.4106%,
width.91231. Referenceloss2.5355,L1=.62355,retained73.0708%,width.90879.
Both zero bucket signs (not a statement of coordinate sign correctness).
Full checkpoints, old/newcomparison andtimings in uv_bucket_split_results.md.
Final independent review approved currentrun; added unsupported cost_target
guard afterruns toprevent silentlyignoring splitflag in thatdifferent solver.
No production math changes afterruns. New loss remains opt-in, no improvement
claim basedonzero signs. Historical LR default remains high.
