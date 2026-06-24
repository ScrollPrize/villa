#!/usr/bin/env node
/**
 * Build-time data generator for the /data_browser page.
 *
 * Produces a hybrid, PUBLIC-ONLY dataset by merging two sources:
 *   1. FACTUAL fields regenerated live from the public S3 metadata.json
 *      (https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json)
 *      — scan/volume/segment counts, pixel sizes, energies, scan locations, etc.
 *   2. A human-editable CURATED overlay checked into the repo
 *      (src/data/atlasOverlay.json) — display names, notes, readings, content,
 *      progress, stages, plus the dashboard / timeline / general copy.
 *
 * EMBARGO: PHerc1667 and PHerc0139 have recovered-text/ink READINGS that are
 * embargoed (Naples, 2026-06-25). Their geometry + factual metadata are public,
 * but their readings (status + images) are stripped to `null` here so they are
 * never published. The overlay marks them with `readings.embargoed === true`.
 *
 * Output: static/data_browser/index.json
 *
 * Run directly (Node 20+ required for global fetch / AbortController):
 *   node scripts/genAtlasData.js
 *
 * Mirrors the style of scripts/genImageDimensions.js and
 * scripts/fetchLatestPosts.js — CommonJS, no extra npm deps.
 */
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const OVERLAY_PATH = path.join(ROOT, "src", "data", "atlasOverlay.json");
const MESH_MANIFEST_PATH = path.join(
  ROOT,
  "static",
  "img",
  "data_browser",
  "meshes",
  "manifest.json"
);
const OUT_DIR = path.join(ROOT, "static", "data_browser");
const OUT_PATH = path.join(OUT_DIR, "index.json");

const S3_METADATA_URL =
  "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json";
const FETCH_TIMEOUT_MS = 25000;

// Scrolls whose READINGS are embargoed — never publish their readings.
const EMBARGOED = new Set(["PHerc1667", "PHerc0139"]);

function readJsonIfExists(p) {
  try {
    if (fs.existsSync(p)) return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch (e) {
    console.warn(`[atlas] failed to read ${path.relative(ROOT, p)}: ${e.message}`);
  }
  return null;
}

async function fetchMetadata() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(S3_METADATA_URL, { signal: controller.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

function uniqSortedNums(arr) {
  return [...new Set(arr.filter((v) => v !== null && v !== undefined))].sort(
    (a, b) => a - b
  );
}
function uniqStrings(arr) {
  return [...new Set(arr.filter((v) => v !== null && v !== undefined && v !== ""))];
}

// Pull {px, energy, loc, name} out of a single scan record. The public
// metadata stores location only under `creation.metadata`, while energy /
// pixel size live both there and (sometimes) under `properties`. Be tolerant.
function scanFacts(scan) {
  const cm = (scan && scan.creation && scan.creation.metadata) || {};
  const props = (scan && scan.properties) || {};
  const px = cm.pixel_size_um ?? props.pixel_size_um ?? null;
  const energy = cm.energy_keV ?? props.energy_keV ?? null;
  const loc = cm.location ?? props.location ?? null;
  const name = (scan && scan.long_id) || null;
  return { px, energy, loc, name };
}

// Derive the FACTUAL fields for one sample from its S3 record.
function deriveFacts(sample) {
  const sampleProps = (sample.sample && sample.sample.properties) || {};
  const scans = sample.scans || {};
  const volumes = sample.volumes || {};
  const segments = sample.segments || {};

  const scanList = Object.values(scans).map(scanFacts);
  const pxVals = scanList.map((s) => s.px).filter((v) => v !== null && v !== undefined);

  return {
    type: sampleProps.type ?? "?",
    desc: (sample.sample && sample.sample.description) || "",
    legacy: sampleProps.legacy_data_url ?? null,
    n_scans: Object.keys(scans).length,
    n_volumes: Object.keys(volumes).length,
    n_segments: Object.keys(segments).length,
    min_px: pxVals.length ? Math.min(...pxVals) : null,
    energies: uniqSortedNums(scanList.map((s) => s.energy)),
    locations: uniqStrings(scanList.map((s) => s.loc)),
    scans: scanList,
  };
}

// Merge curated overlay fields onto a scroll, applying the embargo strip.
function curatedFields(overlayScroll, id) {
  const o = overlayScroll || {};
  let readings = o.readings ?? null;
  // Embargo: never publish readings for embargoed scrolls (or when the overlay
  // flags them, or when readings are missing for an embargoed id).
  if (EMBARGOED.has(id) || (readings && readings.embargoed === true)) {
    readings = null;
  }
  return {
    display: o.display ?? id,
    repository: o.repository ?? null,
    note: o.note ?? "",
    photo: o.photo ?? null,
    bucketUrl: o.bucketUrl ?? null,
    content: o.content ?? null,
    progress: o.progress ?? null,
    stages: o.stages ?? null,
    readings,
  };
}

function generate() {
  const overlay = readJsonIfExists(OVERLAY_PATH) || {};
  const overlayScrolls = overlay.scrolls || {};
  const meshManifest = readJsonIfExists(MESH_MANIFEST_PATH) || {};

  return fetchMetadata()
    .then((meta) => {
      const samples = (meta && meta.samples) || {};
      const ids = uniqStrings([
        ...Object.keys(samples),
        ...Object.keys(overlayScrolls),
      ]);

      const scrolls = ids.map((id) => {
        const facts = samples[id] ? deriveFacts(samples[id]) : null;
        const curated = curatedFields(overlayScrolls[id], id);
        const mesh = meshManifest[id] || null;

        // Prefer S3 for factual fields; fall back to nothing for overlay-only
        // ids without S3 facts (still emit with what's available).
        return {
          id,
          // factual (S3) — null fields where S3 has no record for this id
          type: facts ? facts.type : "?",
          desc: facts ? facts.desc : "",
          legacy: facts ? facts.legacy : null,
          n_scans: facts ? facts.n_scans : 0,
          n_volumes: facts ? facts.n_volumes : 0,
          n_segments: facts ? facts.n_segments : 0,
          min_px: facts ? facts.min_px : null,
          energies: facts ? facts.energies : [],
          locations: facts ? facts.locations : [],
          scans: facts ? facts.scans : [],
          hasS3: !!facts,
          // curated (overlay)
          ...curated,
          // assets
          mesh,
        };
      });

      // Sort by progress.score desc (missing score -> 0).
      scrolls.sort((a, b) => {
        const sa = (a.progress && typeof a.progress.score === "number" && a.progress.score) || 0;
        const sb = (b.progress && typeof b.progress.score === "number" && b.progress.score) || 0;
        return sb - sa;
      });

      const out = {
        updated: (overlay.dashboard && overlay.dashboard.updated) || null,
        _general: overlay._general || "",
        dashboard: overlay.dashboard || null,
        timeline: overlay.timeline || [],
        scrolls,
      };

      fs.mkdirSync(OUT_DIR, { recursive: true });
      fs.writeFileSync(OUT_PATH, JSON.stringify(out, null, 2) + "\n");
      const withMesh = scrolls.filter((s) => s.mesh).length;
      const withReadings = scrolls.filter((s) => s.readings).length;
      const withContent = scrolls.filter((s) => s.content).length;
      console.log(
        `[atlas] wrote ${scrolls.length} scrolls to ${path.relative(ROOT, OUT_PATH)} ` +
          `(${withMesh} mesh, ${withReadings} readings, ${withContent} content)`
      );
    })
    .catch((err) => {
      console.warn(`[atlas] S3 metadata fetch failed: ${err.message}`);
      if (fs.existsSync(OUT_PATH)) {
        console.warn(
          `[atlas] reusing existing ${path.relative(ROOT, OUT_PATH)} (build continues)`
        );
        return;
      }
      console.warn(
        `[atlas] no existing ${path.relative(ROOT, OUT_PATH)} to fall back to; writing overlay-only data`
      );
      // Last resort: emit overlay-only data so the build does not crash.
      const ids = Object.keys(overlayScrolls);
      const scrolls = ids
        .map((id) => ({
          id,
          type: "?",
          desc: "",
          legacy: null,
          n_scans: 0,
          n_volumes: 0,
          n_segments: 0,
          min_px: null,
          energies: [],
          locations: [],
          scans: [],
          hasS3: false,
          ...curatedFields(overlayScrolls[id], id),
          mesh: meshManifest[id] || null,
        }))
        .sort((a, b) => {
          const sa = (a.progress && a.progress.score) || 0;
          const sb = (b.progress && b.progress.score) || 0;
          return sb - sa;
        });
      const out = {
        updated: (overlay.dashboard && overlay.dashboard.updated) || null,
        _general: overlay._general || "",
        dashboard: overlay.dashboard || null,
        timeline: overlay.timeline || [],
        scrolls,
      };
      fs.mkdirSync(OUT_DIR, { recursive: true });
      fs.writeFileSync(OUT_PATH, JSON.stringify(out, null, 2) + "\n");
      console.log(`[atlas] wrote ${scrolls.length} overlay-only scrolls (no S3 facts)`);
    });
}

module.exports = { generate };

if (require.main === module) {
  generate();
}
