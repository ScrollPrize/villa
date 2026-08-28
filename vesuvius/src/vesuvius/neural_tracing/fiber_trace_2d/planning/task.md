# Task: penalize split-piece Defect boundaries

Add a configurable `piece_break_cost` to winding BP. Charge it exactly once
when consecutive pieces from the same original trace have different activity
states: one is active H/V and the other is Defect. Do not charge active-active
or Defect-Defect pairs. Run the established 512-piece benchmark and compare
the positional Defect collapse against the zero-cost baseline.
