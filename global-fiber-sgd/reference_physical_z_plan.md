# Matched physical-Z visualization

Rerender CCW/reference with shared physical base-Z range0–80000 and1000pixel
height: y=(Z-0)*999/80000. Preserve physical horizontal U scale, palette,+11offset,
2px curves, LZW TIFF/SVG and read-only projection reuse. No new solve/projection.

Use original fiber XYZ-Z for fitted curves. On reference placement use the
projected surface Z sampled at cached reference coordinates. Reference background
must also be remapped to physical Z, leaving blank top/bottom coverage rather
than stretching the flattened row coordinate. Rasterize valid adjacent surface
row intervals per column; do not bridge invalid vertices. Keep old UV mode intact.

## Spec update
Optional physical_z_range_base selects diagnostic U-versus-physical-Z, explicitly
not optimized U/V. Store range and exact pixel mapping in output summary.

## Docs/tests
Test shared endpoint pixel mapping, interval raster validity/holes and legacy
mode. Record output paths and shared pixel row coverage in task log/results.
