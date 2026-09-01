# Task: zero-aware winding weight refinement

Extend the iterative five-class winding-weight search so every coordinate
tests zero in addition to its `/2` and `*2` neighbors. Zero is a reversible
candidate, not a permanently disabled coordinate: later search iterations
must be able to re-enable it after other weights change.

Run the corrected search under scale-first semantics from `1,2,2,2,1` on the
established 1024 reference benchmark. Report every accepted move and the final
local optimum.
