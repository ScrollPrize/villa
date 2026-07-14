# NML Fiber Loading With Affine Volume Transforms Plan

## Findings

- `fiber_trace_2d.loader` currently resolves `fiber_paths` / `fiber_glob` and
  calls `load_vc3d_fiber`, so every loaded record is a single `Vc3dFiber`.
- The current specs require VC3D fiber JSON input. NML support is a spec change,
  not just a new glob pattern.
- Existing `.nml` files are XML with `<thing>` graphs, `<node x/y/z>`, and
  `<edge source/target>` entries. The parser must use edges to order each
  fiber; XML node order is not reliable.
- Existing Vesuvius affine helpers support `transform.json` with
  `p_fixed = M @ p_moving` in XYZ order.
- Lasagna's cross-volume dataset path also supports inline `transform` as a
  3x4 XYZ matrix plus `transform_invert`, then converts to a ZYX affine after
  applying cache/source and target scale factors.

## Implementation

- Add a small fiber-source module or extend `fiber_json.py` with:
  - `load_fiber_file(path, transform=None) -> list[Vc3dFiber]`;
  - existing `.json` files returning the current single `Vc3dFiber`;
  - `.nml` files returning one `Vc3dFiber` per usable `<thing>` path.
- Implement NML parsing with `xml.etree.ElementTree`:
  - parse every `<thing>` with at least two finite nodes;
  - build an undirected graph from `<edge>` entries;
  - require each connected component intended as a fiber to be a simple path
    with exactly two endpoints and no node degree above two;
  - order nodes endpoint-to-endpoint through the graph;
  - reject or skip malformed/branching components with clear diagnostics rather
    than guessing a path through branches;
  - use the ordered node coordinates as both `line_points_xyz` and
    `control_points_xyz` initially, unless a later NML convention identifies a
    smaller control-point subset explicitly.
- Add dataset-level affine parsing:
  - accept `fiber_transform_json` / `fiber_transform_json_path` for Vesuvius
    registration `transform.json`;
  - accept inline `fiber_transform` as a 3x4 or 4x4 XYZ matrix, with
    `fiber_transform_invert`;
  - also accept Lasagna-compatible inline aliases `transform` and
    `transform_invert` only for fiber coordinate input, documented as dataset
    fiber-coordinate transforms;
  - validate that at most one transform source is configured.
- Apply transforms only once, immediately after fiber parsing and before record
  identity, bounds checks, deterministic sample indexing, strip-coordinate
  cache keys, prefetch, or training see the fiber.
  - NML and JSON points are in XYZ.
  - Transform matrices are in XYZ.
  - If the matrix maps old/source/moving coordinates to current/fixed
    coordinates, use it directly.
  - If the config sets invert, invert the homogeneous matrix first.
- Preserve existing base-volume semantics:
  - transformed fiber points are current base-level coordinates;
  - `base_volume_scale` still selects reading level and pixel spacing;
  - Lasagna normals continue to come only from the current
    `lasagna_manifest_path`.
- Update `_load_records` to iterate over all fibers returned by a source path.
  A single NML can therefore contribute multiple `_Record`s while the rest of
  the loader remains unchanged.
- Include transform identity in `fiber_identity` and sample identity keys so a
  transformed NML and an untransformed file cannot collide in strip-coordinate
  caches or deterministic sample keys.
- Add explicit diagnostics:
  - count loaded/skipped NML things/components per source;
  - include path, thing id/name, component index, and transform checksum in
    fatal parse or skip messages;
  - keep out-of-volume CP skipping unchanged after transform application.
- Add a concrete S1A NML example config derived from `loader_example.json`:
  - replace only the normal training `datasets` entry with the S1A NML glob;
  - keep the PHercParis4 base-volume path and Lasagna manifest;
  - keep `test_datasets` unchanged until a separate NML test split exists.

## Spec Update

- Change dataset input specs from "VC3D JSON only" to "VC3D JSON or NML".
- Document that NML points are parsed from edge-ordered simple path components.
- Document that the loader normalizes all fiber sources into `Vc3dFiber` before
  existing sampling code runs.
- Document accepted transform keys and direction:
  `fiber_transform_json`, `fiber_transform_json_path`, `fiber_transform`,
  `fiber_transform_invert`, plus Lasagna-compatible inline `transform` /
  `transform_invert`.
- Document that the transform applies to fiber coordinates only and must map
  source fiber XYZ coordinates into current base-volume XYZ coordinates before
  bounds checks and cache-key generation.

## Docs Updates

- Update `docs/code_structure.md`:
  - `fiber_json.py` / fiber-source parsing now covers JSON and NML;
  - `loader.py` record construction applies optional fiber coordinate
    transforms before all downstream strip/prefetch/training paths.
- Add a config example showing an NML glob plus affine transform keys.
- Document `configs/loader_example_s1a_nml.json` as the S1A/PHercParis4 NML
  training-data variant of `loader_example.json`.
- Note that Lasagna manifest usage is unchanged: normals are sampled from the
  current manifest after coordinates are transformed.

## Testing

- Unit-test NML parsing:
  - unordered XML nodes plus edges produce ordered XYZ line/control points;
  - multiple `<thing>` paths in one NML produce multiple `Vc3dFiber` records;
  - branching or disconnected invalid components are rejected/skipped with a
    useful reason.
- Unit-test affine application:
  - inline translation/scale matrix maps NML XYZ points before bounds checks;
  - `fiber_transform_invert` uses the inverse matrix;
  - Vesuvius `transform.json` parsing matches `p_fixed = M @ p_moving`.
- Loader regression tests:
  - JSON fiber config remains unchanged;
  - NML dataset entry loads through `FiberStrip2DLoader` using the existing fake
    zarr/Lasagna manifest helpers;
  - deterministic sample order and sample count include NML-derived fibers;
  - transformed and untransformed inputs produce distinct identity/cache keys.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

- After implementation, add a 2026-07-14 changelog entry for NML fiber input
  and dataset-level affine fiber-coordinate transforms.

## Open Checks Before Coding

- Confirmed from `fibers_s1a_00497z_01497y_03997x_256_v00.nml` that coordinates
  are absolute source-scan coordinates matching the NML bounding boxes, not
  local cube coordinates.
- Confirm the actual old-s1-to-current transform file/key in the local training
  config. If it is the Lasagna inline `transform` form, use that exact spelling
  in the example config.
