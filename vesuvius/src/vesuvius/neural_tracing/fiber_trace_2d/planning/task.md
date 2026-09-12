# Three EMA tracing evaluations during live training

Evaluate current combined training's current EMA0.999, best EMA0.999, then best EMA0.99, in that order. Freeze all three before evaluations and record exact checkpoint steps/samples/hashes. Keep training alive; only manager-exclusive evaluation grants may pause it. No stop/relaunch or manual pause commands.

After each evaluation report comparison against Classic and six earlier NML/JSON/combined600k raw/EMA evaluations plus completed new entries. Every scroll has20% weight in the overall arithmetic mean score. Omit pooled overall scores. Show full per-scroll distributions, highlighting best/tied values per metric.
