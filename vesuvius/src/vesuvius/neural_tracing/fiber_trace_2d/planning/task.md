# Fiber Presence Head And Z-Selected Embedding Supervision

Plan two training changes:

- Add an explicit sheet/fiber presence output head. Supervise all transformed
  control-point pixels as positive `1.0`, and all other reachable valid pixels
  as negative `0.0`. Balance the positive and negative terms to equal overall
  weight. Ignore unreachable edge pixels for the negative term using the same
  shift-reachable region used by contrastive negatives.
- Modify embedding training according to the z-search training todo: when
  multiple strip-z offsets are enabled, do not supervise all same-fiber CP
  offsets as mutually positive. For each CP, choose the already closest/best
  matching other CP plus z-offset in the same fiber group and supervise only
  that positive pair.

Keep the existing direction output and Lasagna ambiguous two-channel direction
encoding unchanged.
