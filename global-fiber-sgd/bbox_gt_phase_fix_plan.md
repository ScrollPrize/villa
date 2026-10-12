# Shared-phase coarse winding correction

Fix bucket sign/distance comparisons by adding the optimized shared phase
increment at B minus that at A to the fixed geometric event correction. The
original anchor phases are already represented by the fixed event correction;
do not add them twice. Keep per-bucket fine offsets and fitted widths excluded.
Use the same coarse quantity in training, fixed sign charge and diagnostics.
Version with bucket_shared_phase=true; historical configs retain old behavior.
Rerun seven intact non-rejectable GT fibers in v3, same params/10080 updates.
Test phase gradients, common-offset invariance, readout equality and old mode.
Independent review before execution. Update results/log/status, preserve v2.

## Spec / docs
Coarse topology includes optimized shared phase; no rounding/detaching that phase.
Fine geometry remains separately weighted. Record precise formula and comparison.
