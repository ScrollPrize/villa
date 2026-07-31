# Remove Unpublished Fiber Version 2

- Remove support for top-level `vc3d_fiber` version 2 because it was never
  published.
- Keep legacy version 1 numeric control-point files readable.
- Keep version 3 as the only object-control-point and segment-descriptor file
  format.
- Remove the obsolete pre-v3 segment metadata schemas and migrations.
- Do not confuse the removed file version with version 3's current
  `tracer_version: 2` field.
