# Task: regular-tracer fiberlet loss and napari colormap selector

Replace the fiberlet DP's simplified additive presence/direction objective with
the regular native fiber tracer's multiplicative local alignment loss. Keep the
fiberlet-specific integer graph, edge-length integration, finite invalid-data
bridges, normal-aware direct smoothness, and the previously agreed omission of
cumulative-history smoothness.

Remove the mistakenly added napari green/red loss endpoint controls. Add a
colormap selector for fiberlet quality instead, with red-yellow-green as the
default and napari's available colormaps as alternatives. Napari is the only
supported viewer for these experimental path artifacts: remove MTL/material
colors and let napari own display color. Visualization changes must not modify
serialized metrics.
