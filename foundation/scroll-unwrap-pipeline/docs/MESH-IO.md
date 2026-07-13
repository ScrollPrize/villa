# Mesh IO Conventions (binding contract)

All mesh bytes on disk go through `scrollkit.io` — a strict, convention-pinned
reader/writer pair. Third-party libraries are in-memory engines only, fed
explicit numpy arrays, never file paths.

## Precision
- Vertex/normal/UV data is float32 end to end; OBJ floats are written with
  `%.9g`, which round-trips float32 exactly.
- Reload-and-compare is the M1 gate: positions, normals and UVs must be
  bit-equal after an OBJ round-trip; faces identical; texture SHA256 equal.

## UV conventions
- PLY (VCGLIB) and OBJ `vt` share a bottom-left origin: no V-flip anywhere.
- The wedge texcoord table is the source of truth. Where per-vertex `s,t`
  equals the wedge table bit-for-bit (groups A/B), compact per-vertex `vt` is
  written; otherwise (group C atlases) wedge UVs are deduplicated bit-exactly
  (uint32 view, no epsilon merging).
- On-screen texture orientation is decided empirically per group by the audit
  (`reports/audit.json: tex_orientation`) — never assumed.

## Library quarantine
- `trimesh`: `Trimesh(vertices=V, faces=F, process=False)` for spatial queries
  only — never its loaders/exporters (default `process=True` merges and
  reorders vertices silently). No Open3D anywhere.
- `pymeshlab`: decimation engine only; inputs are cross-checked array-by-array
  against our reader, outputs extracted as arrays and written by our writer.
  It never writes a file and never touches a texture.
- PyVista/VTK: rendering only; `PolyData` built from our arrays.
- Textures are byte-copied out-of-band and SHA256-verified — never re-encoded.

## Convention traps (CI)
A deliberately asymmetric quad + asymmetric texture fixture turns any silent
V-flip, UV transpose, winding flip, vertex merge/reorder, or material drop into
a hard test failure on every IO path.

## MTL
`map_d` references the same RGBA image as `map_Kd` so viewers honor the
texture's alpha (lacunae render as holes, matching the alpha-test renderer).
