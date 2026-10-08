# Corrected mini5 log

Iteration v3: user also corrected001. Independent plan review passed; retained000
and unchanged settings. Approved benchmark-full command with v3 config completed
in24.1466s including5renders/exports.5restarts(one each), total16.8873mm, pooled
mean2.8146mm.001eligible900→1452.5502vx; failure1400.2883vx. Other4replay objects
exactly match v2. New frozen revision in v3_inputs/manifest.json; old data preserved.
No code changes/rebuild this iteration; actual full mini5 rerun validates config.
Details and command in docs/benchmarks/mini5_corrected_weighted_i4_20260916.md.

2026-09-16: edited piece000 has changed CP span and extrapolated tails. Use native eligibility, preserve edited CP metadata. Version benchmark, preserve original data. Reviewer accepted with length-comparison caveat. Oracle remains paused.

Completed corrected revision under mini5_corrected_weighted_i4_vis_20260916_v2
and adjacent _inputs/manifest.json. Native eligibility1485.5264vs900base voxels.
28.0808s including all5renders;5→5restarts, one each. Other4 replay JSON objects
exactly match previous. Corrected first failure1464.3755base voxels. Total length
15.5618mm, pooled mean2.5936mm. Detailed report docs/benchmarks/mini5_corrected_weighted_i4_20260916.md.
Historical benchmark was versioned rather than overwritten to preserve provenance.
Correction helper explicitly rejects multiple eligible runs; no silent omissions.

Validation: approved native_replay_tests runner passed125native tests and
265Python tests (8skipped) in89.25s; git diff --check clean. Source edit left intact.
