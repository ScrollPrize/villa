# Native 3D Trace2CP Normal-Aware Smoothness Planning

Plan a native 3D Trace2CP smoothness change that uses the available Lasagna
normal to split turn penalties into:

- tangential/surface turn: direction change around the Lasagna normal axis;
- normal/out-of-surface turn: direction change around
  `current_line_dir x Lasagna normal`.

The goal is to allow different smoothness weights for bends that stay on the
local sheet/fiber surface versus bends that tip into/out of the normal
direction.
