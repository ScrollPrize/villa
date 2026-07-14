y reasons the DP is still worse than the original direct traces:

  1. DP is still a coarse lattice path.
     Even with x step 4, y is integer state and transition samples are rounded. The direct tracer is continuous/subpixel and bilinearly samples the direction field every step. This alone  
     can make DP paths look less natural.

  2. The DP scores straight segments, not actual streamline integration.
     A transition from (x0,y0) to (x1,y1) gets one tangent and integrates that against directions along the segment. The direct tracer repeatedly updates direction from the local field, so
     it naturally follows curvature.

  3. Smoothness may still matter too much.
     Even 0.005 * (dy-prev_dy)^2 can be nontrivial when direction losses are small. For debugging, I’d try side DP smoothness at exactly 0 first. If that improves, reintroduce a much      
     smaller penalty.

  4. Rounding inside transition scoring is probably hurting.
     We should replace rounded sample_y / sample_layer lookup with bilinear direction/presence sampling at fractional (x,y,z) along the transition. That would make the DP objective much   
     closer to the direct tracer.

  5. The objective lacks a direct-tracer prior.
     The direct tracer effectively says: “at this point, step along the locally decoded direction.” DP says: “find any monotone path whose segment tangents agree on average.” We can add a 
     stronger local endpoint term, e.g. score direction at both transition start and transition end, or sample more densely along the segment with bilinear interpolation.

  Recommended next fixes, in order:

  1. Set DP smoothness penalties to 0 for side tracing and compare.
  2. Change DP scoring from rounded integer samples to bilinear samples along each transition.
  3. If still worse, add subpixel y lattice, e.g. 0.5 px or 0.25 px state spacing.
  4. Consider a hybrid: use the original direct forward/reverse traces as a corridor, then run DP only inside a band around them. That keeps global connection/scoring while preserving     
     direct-trace behavior.

  The biggest likely win is bilinear fractional scoring + zero/near-zero smoothing.

