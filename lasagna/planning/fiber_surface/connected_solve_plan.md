# Connected deformed solve and visualization

Run 600 joint steps with positive H/V attachment ramp1→20, then200 steps at
constant20 with lower mesh LR. Smoothness1.5, bend5, step0.5; V factor0.1.
Preserve full crossing graph in fitting. Nonexclusive8GiB GPU0 grant (within0–5).
Use approved evaluation runner and manager autostart. Do not bypass manager.

Export an inspection-only crop at fitted winding boundaries14..21 after fitting;
this does not modify the graph or produce a restartable cropped solve. Reuse
existing CT render, JPG CT and8px TIFF overlay/composite. Include H and V labels
in overlay metadata. Validate exact crossing UV equality, finite geometry,
H/V errors and folds, image dimensions and coverage. No claim folds are repaired.

## Spec and docs updates
No loss changes. Document full-fit/central-view distinction, final metrics,
artifact paths, exact commands in task_log/results and changelog. Archive prior log.
