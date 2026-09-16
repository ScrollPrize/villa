# Merge/split single-case benchmark log

User requests corrected mini5v3 case000 as a new benchmark and confirmation of a second touching fiber in edited evaltrace-fiberlet annotation. Original rendering inspected: side-view prediction/trace diverges near terminal region. Edited companion contains CPs and long extrapolated tails and has lost original export provenance; do not assume name proves fiber identity.

Independent plan review passed. Added optional pair_reference benchmark diagnostic,
reusing embedding.nearest_polyline_points and native replay-references; no solver,
sampler or renderer duplication. Companion remains outside scoring. New root
merge_split_case000_20260916_v1, frozen inputs/analysis in pair/. One restart,
replay exactly equal mini5v3 case000,8.33185s including render (not pair analysis).
Geometry corroborates near-contact10.99vx then separation101.66vx, while failed
trace follows companion within3.97vx at90% companion arc. Endpoint exactly on
failed trace, so not independent evidence of physical identity. Detailed evidence,
command and limitations in docs/benchmarks/merge_split_case000_20260916.md.

Approved native_replay_tests:125native tests,266Python passed/8skipped in87.68s.
Distance projection/endpoints regression passed; actual pair run verifies native
eligibility and frozen hashes. git diff --check clean. No global-assignment solver
implemented; requested dataset and geometric confirmation are complete.
