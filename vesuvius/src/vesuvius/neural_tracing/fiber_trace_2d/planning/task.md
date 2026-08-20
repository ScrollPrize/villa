# Task: exact float-cache fiberlet replay

Make the unpublished float32 anchor/fiberlet processing cache transparent to
fiberlet graph replay.

- Cached and eager graph sources must expose identical anchor positions,
  fiberlet geometry, endpoint steps, edge-cost components, transition
  eligibility, transition costs, beam choices, replay geometry, and failures.
- Persist canonical interpolated prediction/normal scoring once per anchor.
- Persist exact nonzero endpoint steps and full edge-cost components per
  fiberlet prefix; connectivity lookahead must not require route payloads.
- Reconstruct route geometry with the same duplicate-point suppression as the
  original DP result.
- Replace the strict unpublished payload schema directly. Do not add a legacy
  reader or repair path.
