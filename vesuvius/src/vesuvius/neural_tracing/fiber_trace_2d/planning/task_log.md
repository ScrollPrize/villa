# Direction Visualization Image-Space Augmentations Log

Current task: add explicit pixel-perfect image-space augmentations to `--dir-vis`
and concatenate the augmented direction overlays into one output image.

Implementation notes:

- Added `_dir_vis_image_space_augmentations` for identity, flip-x, flip-y,
  rot90, rot180, and rot270 variants of the center strip patch and valid mask.
- Materialized every variant as a contiguous NumPy array so flipped/rotated
  negative-stride views do not reach PyTorch.
- Updated `--dir-vis` to run model inference and draw a direction overlay for
  every variant.
- The dir-vis export path is now split into explicit phases:
  `augmented_inputs`, `predictions`, then rendered visualization panels.
- Added `--dbg-dirs`, which adds a second row to `dir_vis.jpg`: the first cell
  is the raw unaugmented patch without arrows, and the remaining cells run
  inference on transformed patches whose center region is overwritten with the
  unaugmented half-image-sized center crop.
- The debug pasted-center side length is half of the native square dir-vis
  image side, rounded up for odd dimensions.
- Dir-vis now center-crops non-square loaded center patches to the largest
  native square before image-space flips/rotations, with no resizing.
- Model inference still runs on each native-resolution image-space variant; the
  4x patch scaling is nearest-neighbor `np.repeat` display scaling only.
- The valid mask gates model normalization and arrow placement only; display
  pixels are clipped directly from the augmented image instead of being blacked
  out by the mask.
- Writes the labeled overlays as one natural-size horizontal `dir_vis.jpg` strip
  with one shared top label band instead of the generic fixed-cell contact sheet,
  so rotated panels remain visibly rotated and are not padded inside equal-sized
  cells.
- Kept the existing 8x8 display-cell, 6-pixel anti-aliased direction segment
  rendering for each panel.
- Updated summary metadata, specs, and code-structure docs.

Validation:

- Compile check:
  `AGENTS_AGENT_MODE=1 PYTHONPATH=vesuvius/src:. python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  passed.
- Focused dir-vis tests:
  `AGENTS_AGENT_MODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k "dir_vis or labeled_panel_grid or receptive_field"`
  passed: `8 passed, 106 deselected in 1.92s`.
- Full fiber 2D loader/runner helper test module:
  `AGENTS_AGENT_MODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: `114 passed in 4.23s`.
- Diff whitespace check:
  `git diff --check -- vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed.
