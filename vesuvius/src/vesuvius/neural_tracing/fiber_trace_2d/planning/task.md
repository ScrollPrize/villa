# Short training comparison

Stop current S1a EMA training at approximately600000 samples. Train fresh five-scroll JSON-only to600000, then combined five-scroll JSON+S1a NML to500000. Combined sampling: every scroll20%; Paris4 JSON10% and NML10%, distance-proportional within each group. Select best raw and best EMA by test losses for each run; evaluate all six on locked50fiber bidirectional inference/tracing, then resume most promising run. Use manager and approved launcher.
