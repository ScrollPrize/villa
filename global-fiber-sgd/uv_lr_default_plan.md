# Adopt measured high LR for future tests

Set the solver's omitted-state-LR default to 0.07348469228349533; retain explicit
historical config values and width LR2. This is the measured 10x LR at global
batch24576, not an instruction to multiply arbitrary historical LRs by ten.
Document high-LR templates for future experiments. Validate default and override.

Spec update: only omitted state LR changes. No loss or initialization changes.
Docs updates: README, task log, changelog; preserve historical results.
Independent review before commit. No GPU rerun needed for this default change.

Next proposed experiment (not part of this default-only commit): separate
width-independent winding-index loss (weight1) and existing width-dependent
position loss (weight0.1), retaining joint optimization and class balancing.
Define crossing/phase conventions explicitly before implementing the new loss.
