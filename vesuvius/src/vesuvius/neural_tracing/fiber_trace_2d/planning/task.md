# Task: Iterative H/V consensus growing

Add a separate constraint-labeling mode that does not use HiGHS. Start from a
deterministically selected long, straight fiber assigned H. The primary seed
must be longer than half the nominal crop side and, among eligible fibers,
must be the straightest fiber closest to the crop center. Then repeatedly
choose the unassigned fiber with the strongest constraint connectivity to the
already assigned active solution. Test H, V, and broken for that fiber and add
the lowest-cost choice.

The connectivity priority uses spatial distance in base voxels and constraint
count, not winding distance. Output final H, V, and broken OBJ layers, plus
matching H/V/broken snapshot layers every 10 added fibers through 100 and every
100 fibers thereafter.

Print detailed choice rows for the first 100 assignments and place the full
consensus summary at the end of the command output.
