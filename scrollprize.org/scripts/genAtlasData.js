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
 * The actual merge lives in src/components/atlas/buildIndex.js — a pure,
 * framework-agnostic module shared with the runtime data browser, so the
 * build-time snapshot and the live (metadata.min.json) fetch derive IDENTICALLY.
 * This script owns only IO: fetch metadata, read the overlay/manifest/config,
 * and write the snapshot.
 *
 * EMBARGO: the embargo mechanism lives in buildIndex.js (EMBARGOED set +
 * `readings.embargoed` flag). The Naples embargo lifted 2026-06-25.
 *
 * Output: static/data_browser/index.json — the bundled fallback + build-time
 * SSR/route data. At runtime the browser prefers the fresher S3 metadata.min.json
 * and only falls back to this file if that fetch fails.
 * Also: static/data_browser/renderSizes.json — HEAD content-lengths of the
 * multi-variant downsampled renders (see resolveRenderSizes below), passed to
 * buildIndex as opts.renderSizes here AND imported by the runtime data browser
 * so both paths pick the same primary render per segment.
 *
 * Run directly (Node 20+ required for global fetch / AbortController):
 *   node scripts/genAtlasData.js
 *
 * Mirrors the style of scripts/genImageDimensions.js and
 * scripts/fetchLatestPosts.js — CommonJS, no extra npm deps.
 */
const fs = require("fs");
const path = require("path");
const { buildIndex } = require("../src/components/atlas/buildIndex");

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
const SIZES_PATH = path.join(OUT_DIR, "renderSizes.json");
const DATA_ACCESS_PATH = path.join(ROOT, "src", "data", "atlasDataAccess.json");

const S3_METADATA_URL =
  "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json";
const FETCH_TIMEOUT_MS = 25000;

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

function writeIndex(out) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(OUT_PATH, JSON.stringify(out, null, 2) + "\n");
}

// --- render sizes (multi-variant primary-render selection) ------------------
//
// buildIndex picks each multi-variant ink segment's PRIMARY render by JPEG
// byte size normalized to physical area (orderByContent) — but buildIndex is
// pure (no fetch), so the sizes are collected here and passed in as
// opts.renderSizes, exactly like overlay/meshManifest/rewrites. The map is
// persisted to static/data_browser/renderSizes.json (checked in, like
// index.json) for two reasons:
//   1. the runtime data browser imports it so the LIVE metadata.min.json path
//      derives the identical order — no drift between bundled and live;
//   2. it doubles as a cache: a render URL's bytes never meaningfully change
//      (filenames embed volume/model ids), so a typical rebuild re-HEADs
//      nothing and only brand-new variants cost a request.

// Every downsampled-render URL that participates in a multi-variant choice —
// the `renders`/`alphaRenders` arrays only exist on those segments.
function collectVariantThumbUrls(index) {
  const urls = new Set();
  for (const scroll of index.scrolls || []) {
    for (const seg of scroll.inkSegments || []) {
      for (const v of [...(seg.renders || []), ...(seg.alphaRenders || [])]) {
        if (v.thumbUrl) urls.add(v.thumbUrl);
      }
    }
  }
  return [...urls];
}

async function headContentLength(url) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10000);
  try {
    const res = await fetch(url, { method: "HEAD", signal: controller.signal });
    if (!res.ok) return null;
    const len = Number(res.headers.get("content-length"));
    return Number.isFinite(len) && len > 0 ? len : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

async function mapLimit(items, limit, fn) {
  const out = new Array(items.length);
  let next = 0;
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, async () => {
      while (next < items.length) {
        const i = next++;
        out[i] = await fn(items[i]);
      }
    })
  );
  return out;
}

// Resolve {url: bytes} for every multi-variant render url, reusing the
// checked-in cache and HEADing only unknown urls. Prunes urls that no longer
// participate so the file tracks the live dataset. Failed HEADs are simply
// absent — buildIndex keeps finest-um order for those segments.
async function resolveRenderSizes(urls) {
  const cached = readJsonIfExists(SIZES_PATH) || {};
  const sizes = {};
  const missing = [];
  for (const u of urls) {
    if (typeof cached[u] === "number" && cached[u] > 0) sizes[u] = cached[u];
    else missing.push(u);
  }
  const fetched = await mapLimit(missing, 12, headContentLength);
  let failed = 0;
  missing.forEach((u, i) => {
    if (fetched[i] != null) sizes[u] = fetched[i];
    else failed++;
  });
  // stable key order for reviewable diffs
  const sorted = Object.fromEntries(Object.keys(sizes).sort().map((k) => [k, sizes[k]]));
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(SIZES_PATH, JSON.stringify(sorted, null, 2) + "\n");
  console.log(
    `[atlas] render sizes: ${urls.length} variant renders, ` +
      `${missing.length - failed} fetched, ${failed} unavailable → ` +
      `${path.relative(ROOT, SIZES_PATH)}`
  );
  return sorted;
}

function logSummary(scrolls, suffix = "") {
  const withMesh = scrolls.filter((s) => s.mesh).length;
  const withReadings = scrolls.filter((s) => s.readings).length;
  const withContent = scrolls.filter((s) => s.content).length;
  const withPred = scrolls.filter((s) => s.n_predictions).length;
  console.log(
    `[atlas] wrote ${scrolls.length} scrolls to ${path.relative(ROOT, OUT_PATH)} ` +
      `(${withMesh} mesh, ${withReadings} readings, ${withContent} content, ${withPred} predictions)${suffix}`
  );
}

function generate() {
  const overlay = readJsonIfExists(OVERLAY_PATH) || {};
  const meshManifest = readJsonIfExists(MESH_MANIFEST_PATH) || {};
  const rewrites =
    ((readJsonIfExists(DATA_ACCESS_PATH) || {}).dataAccess || {}).rewrites || [];
  const opts = { overlay, meshManifest, rewrites };

  return fetchMetadata()
    .then(async (meta) => {
      const samples = (meta && meta.samples) || {};
      // Pass 1 (no sizes) only enumerates the multi-variant render urls; pass
      // 2 re-derives with renderSizes so primary-variant selection applies.
      // buildIndex is pure and fast, so the double derivation is negligible
      // next to the metadata fetch.
      const renderSizes = await resolveRenderSizes(
        collectVariantThumbUrls(buildIndex(samples, opts))
      );
      const out = buildIndex(samples, { ...opts, renderSizes });
      writeIndex(out);
      logSummary(out.scrolls);
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
      const out = buildIndex({}, opts);
      writeIndex(out);
      logSummary(out.scrolls, " [overlay-only, no S3 facts]");
    });
}

module.exports = { generate };

if (require.main === module) {
  generate();
}
