# Preserve complete U/V geometry while aligning approximate Z coverage

User clarifies: do not replace each point's V with physical Z. Preserve all original
UV curves and full reference rectangle. Fit one affine V-to-Z mapping for each
layout from existing matched V/physicalZ pairs (reference grid; fitted fibers).
Render all layouts on one union vertical range with identical pixel mapping.
This changes only vertical scale/offset, not local shape or projection selection.
Keep original horizontal physical U scale; document affine V units as approximateZ,
not actual pointwiseZ. Reference rectangle and fitted extents wholly included.

## Spec update
New opt-in vertical_alignment=affine_z; keep old UV and explicitphysicalZ modes.
No clipping, re-solving or re-projecting. TIFF/LZW1000px,opaque2px lines,+11colors.

## Validation/docs
Test affine fit, endpoint coverage including negative slopes, preserve sample
relative V positions. Record transforms/common bounds and new output paths.
Reference-initialized solve proceeds independently; this uses saved10-pass CCW.

## Output discoverability correction
Publish relative TIFF symlinks for each completed solution map in the output
root next to reference_own.tif. Preserve original files and subdirectories;
no copying large images or rerendering. Refuse to overwrite unrelated paths.
Test link targets, idempotence and collision protection; document the paths.
