# Bucket topology plus weaker metric position

## Follow-up: position weight0.5
User requests half-weight position loss instead of0.1. Run identical fresh/ref
10k suites with position_distance_weight0.5; only weight, output roots and
labels change. Same high LR, six exclusive GPUs, dataset, initialization,
bucket sign penalties and all other terms. Independent config review, existing
tests and production compiled preflights; compare loss trends, referenceL1,
within2, retention, width factor, bucket signs and ranges at each1kcheckpoint.
Old/new scalar losses differ in definition; do not rank them directly.
Spec: no code/default change, experimental configs only. Docs: persist full
tables and commands in uv_bucket_half_results.md, update task/log/status.

Task: implement agreed split objective and run fresh/ref-init10k sampled-pass
solves with high default LR on the same positions5–11 subset, six GPUs.

Use opt-in `bucket_position_split=true`, squared shared-anchor objective only,
no local windows. Bucket delta = j-i + fixed_integer_shift + 0.5*(family_b-family_a). No learned phase,
offset, physical U, or width enters bucket sign or bucket distance. Apply
existing signed/unsigned distance convention, sigma and weight1. Existing
coordinate-derived distance term gets position_distance_weight0.1. Strong
sign hinge and fixed sign charge exclusively use bucket delta; selected-sign
safeguard derives selected bucket pair internally, never old selected geometry.
Keep vertical metric, phase/offset priors, continuation, class balance, broken
probability and all joint updates unchanged. Bucket indices are the existing
anchor-relative hypotheses. Independent review caught the need for a fixed
integer seam/endpoint correction: round((phase_b-anchor_turn_b) -
(phase_a-anchor_turn_a) - wrapped_phase_delta), where anchor_turn is
round(raw_anchor_phase - stored_fractional_anchor_phase), avoiding float32
floor errors near integer boundaries. These are immutable
recorded geometry angles, not optimized phase or position. This prevents
anchor boundary crossings from being mistaken for extra winding separation.

Diagnostics must use the same bucket ordering, explicitly label it, and retain
coordinate reference metrics/width factor. Historical configs remain identical.
Tests: hand-calculated pair costs, width/coordinate independence of topology,
position gradients scaled0.1, fixed charge selected choice, diagnostics parity,
legacy tests and production compiled six-rank gradient validation.

Spec update: new opt-in objective, no data/preparation changes. Docs: task,
status, log, README/changelog and complete fresh/ref checkpoint result tables
including total loss, refL1, within2, retention, signs, width factor, ranges.
Use approved runner; no inference or earlier sign/HV stages. Independently
review before running. Raw losses differ from prior objective and must not be
interpreted as comparable optimization scores across formulations.
