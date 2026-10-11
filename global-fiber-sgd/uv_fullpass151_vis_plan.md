# CCW full-pass solve versus reference

Render the saved CCW10-pass solution via existing reference comparison code and
validated projection cache. Preserve1000px high LZW TIFF, equal-axis physical
scale,2px opaque lines and existing palette; also retain SVG outputs.
Use explicit+11color offset to match reported equal-fiber L1, not the older
length-weighted visualization offset. No optimization, GPU, or strong phase.

## Spec update
Optional global_winding_offset overrides visualization alignment; default retains
historical behavior. Recover initial and final q10/q90 only: intermediate states
were not checkpointed. Report this limitation rather than inventing ranges.

## Docs updates
Archive current task log, record output paths, alignment, metrics and validation.
No changelog entry needed for config-only rendering beyond optional alignment.

## Validation
Reference tests, output summary dimensions/segments, TIFF header LZW, saved-state
metrics reproduce prior L1. Initial metrics reconstructed from exact fresh state.
