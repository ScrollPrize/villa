# Task: limit Fiberlet crop trace attempts

Add a `vc_fiber_trace_chunk` argument that limits the number of anchor seed
attempts independently of the accepted-fiber limit. Preserve strongest-first
seed ordering by descending prediction presence.
