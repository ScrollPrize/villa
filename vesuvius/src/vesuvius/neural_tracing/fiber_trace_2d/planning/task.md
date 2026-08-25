# Task: classify crop traces by principal fiber direction

After `vc_fiber_trace_chunk` has traced a crop, estimate the two principal
unoriented directions from the traces' short local steps. Classify every local
step by which principal direction it is closer to, then classify each complete
fiber as direction-1-dominant, direction-2-dominant, or mixed.

Keep the existing OBJ containing every accepted fiber and additionally write
one OBJ for each of the three classifications so the groups can be displayed
independently. For the complete set and every classified set, also write a
separate point OBJ containing the actual seed anchors where those traces began.
