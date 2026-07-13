# scroll-unwrap-pipeline

Production pipeline for Herculaneum scroll wrap meshes: exact PLY→OBJ
conversion, perceptually-gated decimation, ink-overlay baking, and
self-intersection-free unwrap animations rendered to 4K video.

Input: textured wrap segments (binary PLY, per-vertex `x y z nx ny nz s t`,
single seam-free UV chart in physical voxel units) plus ink-prediction images
in the same UV space. Output, per wrap:

- exact OBJ conversion (bit-equal geometry, SHA-verified textures)
- decimated OBJ at the most aggressive keep-fraction that passes perceptual
  gates
- unwrap animation: 240 frames @ 30 fps, plain / ink-overlay / ink-reveal
  variants, 16:9 and 9:16 framings, 4K masters + 1080p derivatives
- a textured OBJ per frame (ink variant as a two-sided thin shell: recto ink,
  verso plain papyrus)
- per-mesh metrics, certificates, and QA keyframes

## Quickstart

```bash
uv sync                                  # Python 3.12 env from uv.lock
uv run pytest                            # fixture suite (IO traps, kinematics)

uv run python scripts/audit.py           # inventory + conventions oracle
uv run python scripts/convert.py         # M1: exact PLY -> OBJ
uv run python scripts/decimate.py        # M2: perceptual keep-ladder
uv run python scripts/bake_overlays.py   # M2.5: carbon-black ink bake
uv run python scripts/make_render_textures.py   # presentation textures (infill)

uv run python scripts/produce.py         # fleet: animate -> frame OBJs -> videos
uv run python scripts/gate_m5.py         # final gate over all deliverables
# (gate_m5 also expects vision-review verdicts; see docs/QUALITY-GATES.md,
#  'Vision review interface' for the file and schema)
```

Single-mesh flow:

```bash
uv run python scripts/animate.py outputs/decimated/A/<stem>/<stem>_decimated.obj
uv run python scripts/render_video.py outputs/anim/<stem> A --variant ink --framing h
uv run python scripts/clearance_cert.py outputs/anim/<stem>   # interpenetration certificate
```

## Pipeline

1. **Audit** — strict inventory of meshes, textures and ink predictions;
   empirical texture-orientation oracle; UV de-normalization scale fit with
   residuals (the charts are SLIM output in voxel units).
2. **Convert (M1)** — header-driven binary PLY reader → `%.9g` OBJ writer;
   bit-exact round-trip is the gate.
3. **Decimate (M2)** — UV-preserving quadric collapse on a keep-fraction
   ladder; per-rung render-SSIM, close-up crops, Hausdorff and UV-integrity
   gates pick the most aggressive safe rung.
4. **Ink bake (M2.5)** — predictions resampled into the texture canvas,
   polarity-normalized, graded as neutral carbon black absorbed into the
   papyrus.
5. **Unwrap (M3)** — rolling-contact kinematics: the wound part moves rigidly
   (zero transit distortion), material lays flat behind a traveling contact
   line, a narrow curvature-adaptive band blends, and a closed-form
   plane-clearance lift makes roll/sheet interpenetration impossible by
   construction. Chart sanitation (injectivity enforcement, junk-tail trim)
   runs before embedding. A per-frame clearance certificate is a hard,
   non-waivable gate; a deformation-gradient fallback ladder covers meshes
   whose charts defeat the rolling model, with clearance-dominant ranking so
   an interpenetrating fallback never outranks a clean rolling path.
6. **Render (M4/M5)** — headless GPU (EGL) alpha-cutout renderer with a
   documentary look (see `docs/RENDER-STYLE.md`), watermark, reveal push-in,
   raw-RGB pipe into x264. Batch staging is signature-keyed (code revision +
   input file signatures), so outputs from different revisions cannot
   silently coexist; gate checks enforce video-newer-than-trajectory.

Specifications: `docs/MESH-IO.md`, `docs/UNWRAP-MATH.md`,
`docs/RENDER-STYLE.md`, `docs/QUALITY-GATES.md`.

## Layout

```
src/scrollkit/     io | decimate | ink | unwrap | metrics | render
scripts/           stage CLIs + gate_m0..m5 + certificates
configs/global.yaml
tests/             synthetic fixtures (convention traps, analytic cylinder)
```

Expected data layout (not part of this tree): `textured_plys/` for the source
PLY groups and the ink-prediction folders referenced by `scripts/audit.py`;
all derived artifacts land under `outputs/` (gitignored).
