# Reference-initialized solve comparison

Render the completed CCW10-pass reference-initialized solve against the reference
with the established comparison renderer and validated projection cache.
Use global color offset0 (the completed solve's measured offset), not the old11.
Preserve full original UV geometry; single affine vertical alignment per layout,
common union bounds, no pointwise physical-Z flattening or cropping.
1000px-high LZW TIFF, opaque2px fibers; keep existing SVG companions.
Both color transfers: fit geometry/reference colors and reference geometry/fit
colors. Include own-colored fit and full reference. Publish relative root links
to the three completed CCW TIFF maps so all four TIFFs appear in one directory.

## Spec update
No optimization, projection, sampling or rendering algorithm changes. New saved
solution and offset only; reuse approved CPU benchmark entrypoint.

## Validation and documentation
Run existing comparison/affine tests; verify successful renderer exit, complete
map entries, common y mapping, TIFF LZW/height, and root links. Record command,
result paths and mapping in task log/status and a results document.
