# Task: Direction-label MILP diagnostic

Use the gradual crop-fiber direction classification as a diagnostic reference.
Classify the stored crop traces, remove every mixed trace, extract the existing
H/V constraints over the retained traces, solve the existing discrete HiGHS
H/V-plus-broken MILP, and compare its result with the initial direction-1/
direction-2 assignment. Print the resulting labeling errors.
