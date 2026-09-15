# Fiber training-data refinement

Implement opt-in persistent GT refinement from detached presence of the current
training model. Reuse updated geometry for train and test. Immutable baseline GT,
40/80 normal-based ellipse tether, small shape penalty, segment quadrature,
semi-implicit tangent-plane updates, 1/8 ellipse step clamp tapered within 1/8
patch width, and resampling after each step. Save updated fibers at snapshots.

User approved launching a managed continuation from raw current.pt of
5scroll_json_eligible_128_sd1_16x8_ema_20260914_105436, with refinement enabled.
