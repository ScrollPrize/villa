# Task

Fix contrastive embedding negative supervision so edge pixels that cannot host
CP-local positive samples under the configured shift augmentation are not sampled
as negatives.

The negative loss currently treats every valid non-CP pixel as an explicit
negative. Because positives only occur in the CP-neighborhood reachable region,
this teaches the embedding head that unreachable patch edges are always
negative. Instead, negative candidates must also be restricted to the reachable
CP-neighborhood region implied by the configured output-space shift range.
