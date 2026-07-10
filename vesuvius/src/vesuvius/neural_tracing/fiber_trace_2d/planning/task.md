# Task: Opt-In Augment-Vis Profiling

Add an augment-visualization profiling flag. Normal `--augment-vis` exports
should not run or print timing measurements by default. When profiling is
requested, augment-vis should run the same sample/augmentation set twice and
print timing tables for both passes so cold first-use costs and warmed costs
can be compared.
