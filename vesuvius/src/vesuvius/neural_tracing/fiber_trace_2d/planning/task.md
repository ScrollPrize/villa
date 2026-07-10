# Task: Augment-Vis Timing Cleanup

Remove global first-use deterministic sample-order setup from the augment-vis
timing table. The table should measure per-sample patch construction and
augmentation work, not one-time state that is reused by later samples.
