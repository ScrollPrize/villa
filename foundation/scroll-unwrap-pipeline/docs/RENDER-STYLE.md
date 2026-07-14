# Render Style (binding contract)

## Mood
Museum-dark documentary. The papyrus is the hero; everything else recedes.
One continuous, legible unwrapping motion, physically plausible, calm pacing.

## Scene
- Background: near-black charcoal `#101114`, subtle radial vignette (corners
  ~25% darker).
- Lighting: soft 3-point rig — warm-neutral key upper camera-left holding
  papyrus midtones around 0.65–0.75, cool fill ~35% of key camera-right, faint
  rim from behind-above for silhouette separation. Lights fixed in world space.
- Material: textured, two-sided, **alpha cutout (MASK, threshold 0.5)** —
  never depth-sorted blending. Raw OpenGL poly-data mappers with ForceOpaque
  and a fragment-shader alpha test give a true cutout with a working z-buffer
  (any texel with alpha<255 would otherwise route the actor to the translucent
  pass, which has no per-fragment self-depth). Bilinear texture sampling with
  mipmaps OFF (mip-averaged alpha erodes the 0.5 cutout at grazing angles).
  Specular minimal: papyrus is matte.
- Two-sided texturing: front/back single-sided actor pair over the same
  polydata, so the recto (inner face of the winding) carries the ink texture
  and the verso stays plain papyrus.

## Render textures
`outputs/render_textures` infill interior-enclosed alpha dropout up to 0.05%
of the papyrus area per component (Voronoi fill + blur + matched grain);
larger interior components are TRUE lacunae and stay holes, as do
boundary-connected bites. Open dropout windows on a wound roll expose the far
wall's recto ink and read as transparency. Source textures stay byte-exact in
conversion deliverables.

## Ink overlay bake
- Base: the mesh's composite texture (RGBA).
- Ink layer: prediction resampled to texture dims (area filter), polarity
  normalized so ink = high.
- Grade: tint NEUTRAL carbon black `#101010` (the composites are neutral gray;
  any warm cast reads as sepia staining). Opacity = smoothstep(ink, lo=0.42,
  hi=0.78) scaled 0.88. **No glow.** Papyrus fibers remain faintly visible
  through ink midtones — never crushed to #000.
- The result must read as carbon writing absorbed into the papyrus, like
  infrared photographs of opened Herculaneum scrolls.

## Viewing side
The text lives on the INNER face of the winding (the side toward the
umbilicus — the papyrological recto of a rolled scroll). The unwrap exposes
that face; cameras view the side of the flat sheet that pointed inward when
rolled (geometric rule in `scrollkit.render.cinematics`). Text rows stay
horizontal in EVERY framing.

## Camera
- Per-aspect framing fit on the union bbox of the whole trajectory + margin —
  nothing ever clips. 16:9: scroll axis horizontal. 9:16: centered, adaptive
  pull-back as the sheet widens. Never orbit during the morph.

## Timeline (30 fps)
- Plain & ink variants: 240 frames (hold 20 / quintic morph 195 / hold 25).
- Reveal: plain frames 0–239 reused, then camera push-in to a tight flat
  framing, cosine image-space crossfade plain→ink between exact stills, hold,
  and a post-reveal dwell with a slow eased push-in (zoom factor capped so
  side margins stay within the action-safe band in both aspects).
- Watermark: subtle production credit bottom-right, warm paper-gray ~55%
  opacity, insets 2.2% of height (bottom) and width (right), composited after
  any image-space zoom so it stays pinned to the frame.

## Encode (ffmpeg)
- 4K masters (3840×2160 / 2160×3840): `libx264 -preset slow -crf 19
  -pix_fmt yuv420p -profile:v high -level 5.1` with bt709 tags and
  `+faststart`, 30 fps.
- 1080p derivatives: Lanczos downscale, `-crf 18 -level 4.2`, same tags.
- Frames are piped raw RGB to ffmpeg; no intermediate PNGs except the QA
  keyframes.

## QA rubric (vision review, 1–5 each; pass = all ≥4, no artifact flags)
1. Artifact-free geometry (no twist, popping, shearing, z-fighting, hole
   flicker; the late-unwrap roll must be fully opaque — any ghosting of
   text/ink through it is an automatic flag).
2. Texture integrity (sharp, unstretched, correct orientation, no mirrored
   text, clean alpha edges; ink reads as carbon black on the inner face only).
3. Framing & composition (subject placed, nothing clipped, margins balanced).
4. Lighting & readability (both states legible, no blown highlights or
   crushed blacks, silhouette separation).
5. Motion & pacing (eases feel natural, holds correct, reveal lands cleanly).
