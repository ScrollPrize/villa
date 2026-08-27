# Task: Joint adaptive-grid winding BP

Replace the repeated winding BP initialization/calibration cycle with one joint
inference process while retaining the current solver as an explicit comparison
mode.

- Keep aligned Lasagna-normal sign resolution as its existing independent
  preprocessing step. It supplies fixed signed winding measurements; it is not
  an H/V orientation solve.
- Jointly infer H/V/Mixed, integer winding, the global interleaved-lattice
  phase and measurement scale, and any component-local ladder-order gauge.
- Represent global calibration with a small adaptive grid over logarithmic
  positive gain and canonical phase. The grid is an absolute sliding window:
  retained cells never change physical meaning, and new cells are exposed only
  when posterior mass presses against a boundary.
- Run one BP process over the joint model. Do not run a separate H/V/Mixed BP,
  calibration multi-starts, alternating calibration reruns, or independently
  solved calibration candidates in the new mode.
- Make the joint adaptive-grid solver the default and retain the existing
  alternating solver behind an explicit CLI mode for comparison.

Implementation is deferred until the intervening user-requested fix is
completed.
