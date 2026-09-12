# Short training comparison

Stop current S1a EMA training at approximately600000 samples. Train fresh five-scroll JSON-only to600000, then combined five-scroll JSON+S1a NML to600000 (original500000 was a user typo). Combined sampling: every scroll20%; Paris4 JSON10% and NML10%, distance-proportional within each group. Select best raw and best EMA by test losses for each run; evaluate all six on locked50fiber bidirectional inference/tracing, then resume most promising run. Use manager and approved launcher.

The initial combined500k comparison already finished and won; evaluate its
continuation at601600 and resume it afterward. Print full per-scroll and pooled
raw/EMA statistics for all three matched-budget conditions, retaining500k as
supplementary history. Best selection includes the parent run, never later-than-
cutoff snapshots.
