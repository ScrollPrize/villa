/**
 * Unit tests for buildIndex() — specifically the multi-volume ink-segment
 * render-variant surfacing (renders[] / alphaRenders[]) added in
 * extractRenderVariants()/toRenderMeta()/groupRendersByVolume(). Exercises the
 * public buildIndex() entry point end-to-end with small inline fixture
 * "samples" objects (shaped like the public S3 metadata.json `samples` map),
 * never the unexported helpers directly.
 *
 * Run: yarn test (or `npm test`), which invokes `node --test` with a glob
 * scoped to this directory. Note a bare directory positional argument
 * (`node --test src/components/atlas`, no glob, no trailing filename) does
 * NOT work on Node v22.22.2 here — Node's CLI treats it as a script to
 * `require()` (MODULE_NOT_FOUND, since there's no index.js) rather than as a
 * directory to search recursively for test files; this reproduces identically
 * in a throwaway directory outside this repo, so it's a Node CLI quirk, not a
 * bug in this file. `node --test` (default discovery, no path) and an
 * explicit file/glob path both work fine — see package.json's "test" script.
 */
const test = require("node:test");
const assert = require("node:assert/strict");
const { buildIndex } = require("./buildIndex");

// Build one `data[]` entry for a segment. `type` defaults to the 2D
// full-resolution ink-detection kind; pass "ink-detection-downsampled",
// "alpha-render", or "alpha-render-downsampled" to build the other kinds.
function mkEntry({
  type = "ink-detection",
  targetVolume,
  modelId,
  modelName,
  renderLevel,
  path,
  root = "s3://vesuvius-challenge-open-data",
}) {
  const d = { type, origins: [{ access_roots: [{ url: root }], path }] };
  const params = {};
  if (targetVolume !== undefined) params.target_volume = targetVolume;
  if (modelId !== undefined) params.model_id = modelId;
  if (renderLevel !== undefined) params.render_level = renderLevel;
  if (Object.keys(params).length) d.parameters = params;
  if (modelName !== undefined) {
    d.creation_info = { provenance: { parameters: { model: modelName } } };
  }
  return d;
}

// Minimal sample scaffold: one scroll id with the given volumes/segments.
function mkSamples(id, { volumes = {}, segments = {} } = {}) {
  return {
    [id]: {
      sample: { properties: {}, description: "" },
      scans: {},
      volumes,
      segments,
    },
  };
}

const OPTS = { overlay: {}, meshManifest: {}, rewrites: [] };

function firstSegment(samples, id) {
  const out = buildIndex(samples, OPTS);
  const scroll = out.scrolls.find((s) => s.id === id);
  assert.ok(scroll, `expected a scroll with id ${id}`);
  assert.ok(scroll.inkSegments.length > 0, `expected at least one ink segment for ${id}`);
  return scroll.inkSegments[0];
}

test("(a) 2 ink-detection entries at distinct target_volumes surface as sorted renders[]", () => {
  const samples = mkSamples("TestScrollA", {
    volumes: {
      volA: { properties: { pixel_size_um: 2.0, energy_keV: 53 } },
      volB: { properties: { pixel_size_um: 4.0, energy_keV: 70 } },
    },
    segments: {
      seg1: {
        suffix: "w010_seg1",
        data: [
          // Deliberately out of order to prove the sort, not insertion order.
          mkEntry({
            targetVolume: "volB",
            modelId: "20250101000000",
            modelName: "modelB",
            path: "segments/seg1/ink-detection/volB_ink.tif",
          }),
          mkEntry({
            targetVolume: "volA",
            modelId: "20250101000000",
            modelName: "modelA",
            path: "segments/seg1/ink-detection/volA_ink.tif",
          }),
        ],
      },
    },
  });

  const seg = firstSegment(samples, "TestScrollA");
  assert.strictEqual(seg.renders.length, 2);
  // Finest (smallest um) first.
  assert.strictEqual(seg.renders[0].targetVolume, "volA");
  assert.strictEqual(seg.renders[0].um, 2.0);
  assert.strictEqual(seg.renders[0].energy, 53);
  assert.strictEqual(seg.renders[0].modelName, "modelA");
  assert.strictEqual(seg.renders[1].targetVolume, "volB");
  assert.strictEqual(seg.renders[1].um, 4.0);
  assert.strictEqual(seg.renders[1].energy, 70);
  assert.strictEqual(seg.renders[1].modelName, "modelB");
  // Legacy top-level fields point at the finest ("one") variant.
  assert.strictEqual(seg.um, 2.0);
  assert.ok(seg.full.endsWith("volA_ink.tif"));
});

test("(b) a single ink-detection variant keeps exact legacy full/down/preview/um semantics and omits renders", () => {
  const samples = mkSamples("TestScrollB", {
    volumes: {},
    segments: {
      seg2: {
        suffix: "w020_seg2",
        data: [
          mkEntry({ path: "segments/seg2/ink-detection/seg2_7.91um_ink.tif" }),
          mkEntry({
            type: "ink-detection-downsampled",
            path: "segments/seg2/ink-detection/seg2_7.91um_ink_ds8.jpg",
          }),
        ],
      },
    },
  });

  const seg = firstSegment(samples, "TestScrollB");
  assert.strictEqual(seg.renders, undefined);
  assert.strictEqual(seg.um, 7.91);
  assert.ok(seg.full.endsWith("seg2_7.91um_ink.tif"));
  assert.ok(seg.down.endsWith("seg2_7.91um_ink_ds8.jpg"));
  assert.strictEqual(seg.preview, seg.down);
});

test("(c) duplicate model runs at the same target_volume collapse to the newest model_id (not 4 entries)", () => {
  const samples = mkSamples("TestScrollC", {
    volumes: {
      volA: { properties: { pixel_size_um: 2.0, energy_keV: 53 } },
      volB: { properties: { pixel_size_um: 4.0, energy_keV: 70 } },
    },
    segments: {
      // Mirrors the real PHerc0172 20251107110950 edge case: 2 target_volumes
      // x 2 model_ids each = 4 raw ink-detection entries.
      seg3: {
        suffix: "20251107110950",
        data: [
          mkEntry({
            targetVolume: "volA",
            modelId: "20250101000000",
            path: "segments/seg3/ink-detection/volA_run1.tif",
          }),
          mkEntry({
            targetVolume: "volA",
            modelId: "20250301000000",
            path: "segments/seg3/ink-detection/volA_run2.tif",
          }),
          mkEntry({
            targetVolume: "volB",
            modelId: "20250115000000",
            path: "segments/seg3/ink-detection/volB_run1.tif",
          }),
          mkEntry({
            targetVolume: "volB",
            modelId: "20250420000000",
            path: "segments/seg3/ink-detection/volB_run2.tif",
          }),
        ],
      },
    },
  });

  const seg = firstSegment(samples, "TestScrollC");
  assert.strictEqual(seg.renders.length, 2);
  const byVol = Object.fromEntries(seg.renders.map((r) => [r.targetVolume, r]));
  assert.strictEqual(byVol.volA.modelId, "20250301000000");
  assert.ok(byVol.volA.url.endsWith("volA_run2.tif"));
  assert.strictEqual(byVol.volB.modelId, "20250420000000");
  assert.ok(byVol.volB.url.endsWith("volB_run2.tif"));
  // Top-level fields follow the finest surviving (newest-run) variant —
  // deterministic newest-wins, the disclosed acceptable behavior change.
  assert.strictEqual(seg.full, byVol.volA.url);
});

test("(d) a target_volume absent from sample.volumes does not throw, falls back to filename um parsing, and has null energy", () => {
  const samples = mkSamples("TestScrollD", {
    volumes: {
      volA: { properties: { pixel_size_um: 2.0, energy_keV: 53 } },
      // Note: no entry for "volMissing".
    },
    segments: {
      seg4: {
        suffix: "w030_seg4",
        data: [
          mkEntry({
            targetVolume: "volA",
            modelId: "1",
            path: "segments/seg4/ink-detection/volA_ink.tif",
          }),
          mkEntry({
            targetVolume: "volMissing",
            modelId: "1",
            path: "segments/seg4/ink-detection/volMissing_9.0um_ink.tif",
          }),
        ],
      },
    },
  });

  let seg;
  assert.doesNotThrow(() => {
    seg = firstSegment(samples, "TestScrollD");
  });
  assert.strictEqual(seg.renders.length, 2);
  const missing = seg.renders.find((r) => r.targetVolume === "volMissing");
  assert.ok(missing);
  assert.strictEqual(missing.um, 9.0);
  assert.strictEqual(missing.energy, null);
});

test("(e) SYNTHETIC: 2 alpha-render entries at distinct target_volumes surface symmetrically as alphaRenders[]", () => {
  const samples = mkSamples("TestScrollE", {
    volumes: {
      volC: { properties: { pixel_size_um: 3.24, energy_keV: 54 } },
      volD: { properties: { pixel_size_um: 7.91, energy_keV: 88 } },
    },
    segments: {
      seg5: {
        suffix: "w040_seg5",
        data: [
          mkEntry({
            type: "alpha-render",
            targetVolume: "volD",
            modelId: "1",
            modelName: "alphaModelD",
            path: "segments/seg5/alpha-render/volD_alpha.tif",
          }),
          mkEntry({
            type: "alpha-render",
            targetVolume: "volC",
            modelId: "1",
            modelName: "alphaModelC",
            path: "segments/seg5/alpha-render/volC_alpha.tif",
          }),
          // A downsampled counterpart is required for an alpha-only segment
          // to be emitted at all (mirrors the 3 alpha-only PHercParis4 segs).
          mkEntry({
            type: "alpha-render-downsampled",
            targetVolume: "volC",
            modelId: "1",
            path: "segments/seg5/alpha-render/volC_alpha_ds.jpg",
          }),
          mkEntry({
            type: "alpha-render-downsampled",
            targetVolume: "volD",
            modelId: "1",
            path: "segments/seg5/alpha-render/volD_alpha_ds.jpg",
          }),
        ],
      },
    },
  });

  const seg = firstSegment(samples, "TestScrollE");
  assert.strictEqual(seg.renders, undefined);
  assert.strictEqual(seg.alphaRenders.length, 2);
  assert.strictEqual(seg.alphaRenders[0].targetVolume, "volC");
  assert.strictEqual(seg.alphaRenders[0].um, 3.24);
  assert.strictEqual(seg.alphaRenders[0].energy, 54);
  assert.strictEqual(seg.alphaRenders[0].modelName, "alphaModelC");
  assert.ok(seg.alphaRenders[0].thumbUrl.endsWith("volC_alpha_ds.jpg"));
  assert.strictEqual(seg.alphaRenders[1].targetVolume, "volD");
  assert.strictEqual(seg.alphaRenders[1].um, 7.91);
  assert.strictEqual(seg.alphaRenders[1].energy, 88);
  assert.strictEqual(seg.alphaRenders[1].modelName, "alphaModelD");
  assert.ok(seg.alphaRenders[1].thumbUrl.endsWith("volD_alpha_ds.jpg"));
});

test("(f) 2D and 3D render-variant counts are computed independently — neither gates the other", () => {
  const samples = mkSamples("TestScrollF", {
    volumes: {
      volA: { properties: { pixel_size_um: 2.0, energy_keV: 53 } },
      volB: { properties: { pixel_size_um: 4.0, energy_keV: 70 } },
    },
    segments: {
      // 2 ink-detection variants, only 1 alpha-render variant.
      segF1: {
        suffix: "w050_segF1",
        data: [
          mkEntry({
            targetVolume: "volA",
            modelId: "1",
            path: "segments/segF1/ink-detection/volA_ink.tif",
          }),
          mkEntry({
            targetVolume: "volB",
            modelId: "1",
            path: "segments/segF1/ink-detection/volB_ink.tif",
          }),
          mkEntry({
            type: "alpha-render",
            targetVolume: "volA",
            modelId: "1",
            path: "segments/segF1/alpha-render/volA_alpha.tif",
          }),
        ],
      },
      // Only 1 ink-detection variant, 2 alpha-render variants.
      segF2: {
        suffix: "w051_segF2",
        data: [
          mkEntry({
            targetVolume: "volA",
            modelId: "1",
            path: "segments/segF2/ink-detection/volA_ink.tif",
          }),
          mkEntry({
            type: "alpha-render",
            targetVolume: "volA",
            modelId: "1",
            path: "segments/segF2/alpha-render/volA_alpha.tif",
          }),
          mkEntry({
            type: "alpha-render",
            targetVolume: "volB",
            modelId: "1",
            path: "segments/segF2/alpha-render/volB_alpha.tif",
          }),
        ],
      },
    },
  });

  const out = buildIndex(samples, OPTS);
  const scroll = out.scrolls.find((s) => s.id === "TestScrollF");
  const segF1 = scroll.inkSegments.find((s) => s.id === "segF1");
  const segF2 = scroll.inkSegments.find((s) => s.id === "segF2");
  assert.ok(segF1);
  assert.ok(segF2);

  assert.strictEqual(segF1.renders.length, 2);
  assert.strictEqual(segF1.alphaRenders, undefined);

  assert.strictEqual(segF2.renders, undefined);
  assert.strictEqual(segF2.alphaRenders.length, 2);
});
